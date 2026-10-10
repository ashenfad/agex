"""Agents, and the sessions they drive.

An :class:`Agent` is a loop: a model (a :class:`~agex.providers.Provider`),
a primer, and the settings it calls the model with. It owns no world. A
session gives it one: ``agent.session(ws)`` drives a nontainer workspace
turn by turn, with its conversation stored in the workspace's branch, so
a checkout rewinds the agent's memory with the files and a fork carries
it.

Each turn is one run inside ``ws.turn``. The model is asked; the tools it
calls run against the workspace (each mutating call committed as it
lands); notes queued mid-turn ride the tool results; and the loop goes
round until the model answers without calling a tool. The turn's end
stores the run (agex's own record, :mod:`agex.record`), settles the
inbox and lands one commit stamped with how the run ended.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from nontainer import NotSupportedError, Profile, SessionRunner, Workspace, conversation
from nontainer.adapters.render import toolkit_instructions
from nontainer.adapters.tools import Tool, Toolset
from nontainer.compaction import Policy
from nontainer.inbox import Inbox, Note
from nontainer.sessions import Sessions, answer_notes
from nontainer.turns import (
    Delivered,
    DeliveredNote,
    RunEnded,
    RunStarted,
    RunStatus,
    ToolEnded,
    ToolStarted,
    TurnEvent,
    Usage,
)
from pydantic_ai.models import Model

from . import compaction
from .providers import Provider, Reply, Settings, ToolSpec
from .providers.pydanticai import PydanticAIProvider
from .record import (
    Message,
    Run,
    Text,
    ToolCall,
    ToolResult,
    dump_run,
    load_run,
    new_id,
)

if TYPE_CHECKING:
    from .task import Task

__all__ = [
    "CLOSING_NOTE",
    "CUT_OFF",
    "HARNESS",
    "Agent",
    "Outcome",
    "RunStream",
    "Session",
    "Status",
    "closing_note",
]

_logger = logging.getLogger(__name__)

HARNESS = "agex"
"""The name agex's conversations are stored under, in the workspace's
conversation index."""

MAX_STEPS = 100
"""How many model calls a run may make before it is stopped."""

CLOSING_NOTE = "[turn ended early:"
"""How the note closing a cancelled or failed run begins: the run keeps
the work it did, and the model reads that it was cut short."""

STOPPED = "stopped"
"""Why a turn :meth:`Session.cancel` stopped ended."""

WOKEN = "(You were woken, and nothing new is waiting.)"
"""The message a woken turn opens with when nothing is waiting after all:
what woke it was taken before the turn began."""

UNANSWERED = "not run: the turn ended before this call started"
"""The result a tool call gets when its turn ended before running it."""

CUT_OFF = (
    "no result: the turn ended while this call was running, and it may "
    "have finished anyway; check the workspace before repeating it"
)
"""The result a tool call gets when its turn ended while it ran: a tool
on its worker thread runs on to its end."""


def closing_note(reason: str) -> Message:
    """The assistant message closing a cancelled or failed run."""
    return Message(
        id=new_id(),
        role="assistant",
        parts=(
            Text(
                text=f"{CLOSING_NOTE} {reason}. The work above this point is "
                "real and done.]"
            ),
        ),
    )


def _delivered(notes: Sequence[Note]) -> Delivered:
    return Delivered(notes=tuple(DeliveredNote.of(n) for n in notes))


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


class _SyncLoop:
    """The event loop the blocking front doors (``say``, ``resume``) run
    their turns on: one for the process, on a thread of its own, made
    when first needed.

    One loop, not one per call: a model's client keeps the connections it
    opened, and they belong to the loop they were opened on, so a turn on
    a fresh loop (``asyncio.run``) would find them on a closed one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                threading.Thread(
                    target=loop.run_forever, name="agex-sync", daemon=True
                ).start()
                self._loop = loop
            return self._loop

    def run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop())
        try:
            return future.result()
        except BaseException:
            # interrupted while waiting (Ctrl-C): the turn is cancelled,
            # and ends cancelled on its loop as any other would
            future.cancel()
            raise


_SYNC = _SyncLoop()


def _block(coro: Any, instead: str) -> Any:
    """Run ``coro`` to its end on agex's own loop, from code that has
    none; refused where a loop is running, which this would block.
    ``instead`` names the awaitable to use there (``session.asay``)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no loop here, so blocking is fine
    else:
        coro.close()
        raise RuntimeError(
            "this blocks, and this thread is running an event loop; "
            f"use `await {instead}(...)` there"
        )
    # outside the handler, so what the run raises isn't chained to it
    return _SYNC.run(coro)


_TURN_LOOP: contextvars.ContextVar[asyncio.AbstractEventLoop | None] = (
    contextvars.ContextVar("agex_turn_loop", default=None)
)
"""The loop the running turn is on, as seen from the work it does: a
tool call and a host call carry the turn's context with them."""


def _on_turn_loop(coro: Any, instead: str) -> Any:
    """Run ``coro`` to its end on the running turn's loop, from host
    code a tool call is running (an agent-defined task's helper, say):
    a model's client keeps its connections on the loop it opened them
    on, and the turn's model opened them on this one. Outside a turn,
    on agex's own loop (:func:`_block`)."""
    loop = _TURN_LOOP.get()
    if loop is None or loop.is_closed():
        return _block(coro, instead)
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        coro.close()
        raise RuntimeError(
            "this blocks, and this thread is running the turn's event loop; "
            f"use `await {instead}(...)` there"
        )
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


Status = Literal[
    "completed", "cancelled", "interrupted", "failed", "success", "needs_input"
]
"""How a run ended: a run status (:data:`nontainer.turns.RunStatus`),
or for a task, ``success`` (it handed back its value) or
``needs_input``."""


@dataclass(frozen=True)
class Outcome:
    """How a run went.

    ``status`` is how it ended, ``text`` the model's last reply, and
    ``head`` the commit the run's workspace stands at afterwards
    (``None`` on a workspace that does not version). ``events`` are what
    the run streamed, and ``message`` says why it did not complete, when
    it did not.

    A task's outcome also carries its ``value`` (on ``success``) and,
    when its world was kept, ``ref``: the session its fork lives on.
    A session's turn ends with a run status; a task's ends ``success``,
    ``failed`` (``task.fail``, an error, or no answer), ``needs_input``
    (``task.needs_input``: ``message`` is the question, and the world is
    always kept), ``cancelled`` or ``interrupted``.
    """

    status: Status
    text: str
    run_id: str
    head: str | None
    events: tuple[TurnEvent, ...] = ()
    message: str | None = None
    value: Any = None
    ref: str | None = None
    _resume: Callable[..., Awaitable[Outcome]] | None = field(
        default=None, repr=False, compare=False
    )

    def resume(self, answer: Any, /, **live_inputs: Any) -> Outcome:
        """Answer the question a ``needs_input`` task asked, and run it
        on to its next end; the new outcome. The in-process shortcut for
        its task's ``resume``: the call's own inputs and world are used
        again, so only a live input to replace is passed. From a
        coroutine, ``await aresume``."""
        return _block(self.aresume(answer, **live_inputs), "outcome.aresume")

    async def aresume(self, answer: Any, /, **live_inputs: Any) -> Outcome:
        """:meth:`resume`, from a coroutine."""
        if self._resume is None:
            raise ValueError(
                f"only a task's needs_input outcome resumes; this one is {self.status}"
            )
        return await self._resume(answer, **live_inputs)


class Agent:
    """A loop over a model.

    ``model`` is a :class:`~agex.providers.Provider`, a model name
    pydantic-ai knows (``"openrouter:meta/muse-spark-1.3-contributor"``),
    or a pydantic-ai ``Model`` (one pointed at a local OpenAI-compatible
    endpoint, say). ``primer``
    opens every request, ahead of the workspace's own tool instructions.
    ``profile`` is the environment for worlds the agent creates itself; a
    session uses its workspace's own. ``max_steps`` bounds the model
    calls in one run, compaction's summary calls among them. ``compaction`` folds a conversation that reaches
    its budget into a summary, within a run as well (see
    :mod:`agex.compaction`); without one, nothing is folded, though a
    world's folds are still sent as recorded.
    """

    def __init__(
        self,
        model: Provider | Model | str,
        *,
        primer: str = "",
        profile: Profile | None = None,
        settings: Settings = Settings(),
        max_steps: int = MAX_STEPS,
        compaction: Policy | None = None,
    ) -> None:
        self.provider: Provider = (
            PydanticAIProvider(model) if isinstance(model, (str, Model)) else model
        )
        self.primer = primer
        self.profile = profile
        self.settings = settings
        self.max_steps = max_steps
        self.compaction = compaction

    def __repr__(self) -> str:
        return f"<Agent {self.provider.name}>"

    def session(
        self,
        ws: Workspace,
        *,
        inbox: Inbox | None = None,
        sessions: Sessions | bool | None = None,
    ) -> Session:
        """A session driving ``ws``, which may already hold an agex
        conversation (it continues) or none (it starts one).
        ``sessions`` lets it delegate; see :class:`Session`."""
        return Session(self, ws, inbox=inbox, sessions=sessions)

    def task(self, fn: Callable[..., Any]) -> Task:
        """A task: ``fn``'s signature and docstring, run by this agent on
        a world of its own (see :mod:`agex.task`). Use it as a
        decorator."""
        from .task import Task

        # the decorating code's names: a type defined in the same
        # function resolves under postponed annotations
        return Task(self, fn, names=dict(sys._getframe(1).f_locals))


class Session:
    """An agent driving one workspace, turn by turn.

    ``say`` runs a turn and returns its :class:`Outcome`; ``asay`` is the
    same from a coroutine; ``stream`` starts a turn and hands back its
    events as they happen (a :class:`RunStream`). ``inbox`` takes notes
    mid-turn: each lands on the next tool result.

    A turn is a task of its own, so it belongs to the session, not to
    whoever watches it: leaving a stream early stops watching, and the
    turn goes on to its end; ``cancel`` stops it. One turn runs at a
    time; ``running`` says whether one is.

    ``sessions`` lets the agent delegate, through the ``sessions`` tool:
    ``True`` builds a helper whose delegates are sessions of this same
    agent (:class:`~agex.delegation.Runner`), or pass a
    :class:`nontainer.sessions.Sessions` of your own. Answers that land
    mid-turn ride the next tool result, as a note in ``inbox`` does; one
    that lands between turns is waiting for the next, or for ``wake``,
    a turn that opens with what is waiting. A helper the session builds
    is built by its first turn and runs the delegates on that turn's
    loop (agex's own, for ``say``), so they share the parent's; a later
    turn on another loop is refused. ``close`` closes it.

    A turn ends ``completed``, ``cancelled``, ``failed`` (an error, or
    ``max_steps`` model calls) or ``interrupted`` (a provider error worth
    retrying, see :meth:`~agex.providers.Provider.transient`), which
    ``resume`` continues in place. A cancelled or failed turn keeps its
    work with a closing note, so the model remembers what it did; none
    of the endings raise.

    A workspace whose conversation another harness wrote is refused: the
    runs it holds are in that harness's format.
    """

    def __init__(
        self,
        agent: Agent,
        ws: Workspace,
        *,
        inbox: Inbox | None = None,
        sessions: Sessions | bool | None = None,
    ):
        index = conversation.index_of(ws)
        if index is not None and index.harness != HARNESS:
            raise NotSupportedError(
                f"session {ws.session!r} holds a conversation {index.harness!r} "
                "wrote; agex reads only its own"
            )
        self.agent = agent
        self.ws = ws
        self.inbox = inbox if inbox is not None else Inbox()
        # a helper the session builds is built by its first turn, on that
        # turn's loop; its loop is also the sign that closing it is the
        # session's to do
        self._runner: SessionRunner | None = None
        if sessions is True:
            from .delegation import Runner, _as_runner

            # made now, so a world that can't delegate is refused now
            self._runner = _as_runner(Runner(agent, ws))
        self._sessions_loop: asyncio.AbstractEventLoop | None = None
        self.sessions: Sessions | None = None
        """The helper the agent delegates through, if it can: one built
        for ``sessions=True`` exists from the first turn on."""
        self._use(sessions if isinstance(sessions, Sessions) else None)
        self.last: Outcome | None = None
        """The outcome of the last turn that ended."""
        self._in_flight: set[asyncio.Task[Outcome]] = set()
        self._cancel_requested = False
        self._live = False

    def __repr__(self) -> str:
        return f"<Session {self.ws.session!r} of {self.agent!r}>"

    @property
    def running(self) -> bool:
        """Whether a turn is running."""
        return any(not task.done() for task in self._in_flight)

    def _track(self, task: asyncio.Task[Outcome]) -> asyncio.Task[Outcome]:
        # asyncio keeps only weak references to tasks: a turn nobody
        # watches must still run to its end, so the session holds it
        self._in_flight.add(task)
        task.add_done_callback(self._settled)
        return task

    def _settled(self, task: asyncio.Task[Outcome]) -> None:
        self._in_flight.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # retrieved here, so a turn nobody watched does not log
            # "exception was never retrieved"; the stored run says how
            # it ended
            _logger.debug("turn ended with %r", task.exception())

    def _use(self, sessions: Sessions | None) -> None:
        self.sessions = sessions
        self.toolset = Toolset(self.ws, vision=False, sessions=sessions)
        self._tools: dict[str, Tool] = {t.name: t for t in self.toolset.tools()}
        self._specs = [ToolSpec.of(t) for t in self._tools.values()]

    def _bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Ready a turn about to start on ``loop``: a helper the session
        builds is built here, on the loop its first turn runs on, so the
        delegates share the parent's loop and with it the provider's
        connections, which belong to the loop that opened them."""
        if self._runner is None:
            return
        if self._sessions_loop is None:
            self._use(Sessions(self.ws, self._runner, loop=loop))
            self._sessions_loop = loop
        elif loop is not self._sessions_loop:
            raise RuntimeError(
                "this session's delegates run on the event loop of its first "
                "turn, and this turn is on another; drive the session from one "
                "loop (say() runs on agex's own)"
            )

    def close(self) -> None:
        """Close the ``sessions`` helper the session built, which waits
        for its delegates' runs; one passed in is its owner's to close.
        From a coroutine, ``await aclose``."""
        loop = self._sessions_loop
        if loop is None or self.sessions is None:
            return
        try:
            here: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            here = None
        if loop is here:
            self.sessions.close()  # refused while runs are pending: aclose
        else:
            asyncio.run_coroutine_threadsafe(self.sessions.aclose(), loop).result()

    async def aclose(self) -> None:
        """:meth:`close`, from a coroutine."""
        loop = self._sessions_loop
        if loop is None or self.sessions is None:
            return
        if loop is asyncio.get_running_loop():
            await self.sessions.aclose()
        else:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(self.sessions.aclose(), loop)
            )

    # -- the stored conversation ---------------------------------------------------

    @property
    def runs(self) -> list[Run]:
        """The runs the workspace holds, oldest first."""
        kv = self.ws.provider.kv
        index = conversation.read_index(kv)
        if index is None:
            return []
        bodies = conversation.read_runs(kv, index.runs, index)
        return [load_run(bodies[rid]) for rid in index.runs if rid in bodies]

    def _system(self) -> Message:
        instructions = toolkit_instructions(
            self.ws,
            split=self.toolset.split,
            turn_commits=not self.ws.autocommit,
        )
        text = "\n\n".join(t for t in (self.agent.primer, instructions) if t)
        return Message(id="system", role="system", parts=(Text(text=text),))

    # -- where a run may end (a task's loop overrides these) ------------------------

    def _after_call(self, output: str, is_error: bool) -> tuple[str, bool]:
        """A tool call's result as the model will read it."""
        return output, is_error

    def _done(self) -> bool:
        """Whether the run is over once the call just made returns,
        leaving any calls after it in the same reply unrun."""
        return False

    def _on_stop(self, reply: Message) -> Message | str | None:
        """What follows a reply that calls no tool: ``None`` ends the run
        completed, a message is sent back to the model and the run goes
        on, and text fails the run, saying why."""
        return None

    # -- the front doors ------------------------------------------------------------

    def say(self, prompt: str) -> Outcome:
        """Run one turn; its outcome. From a coroutine, ``await asay``."""
        return _block(self.asay(prompt), "session.asay")

    async def asay(self, prompt: str) -> Outcome:
        """Run one turn; its outcome."""
        return await self.stream(prompt).wait()

    def resume(self) -> Outcome:
        """Continue the last turn in place, which an interruption cut
        short; its outcome. From a coroutine, ``await aresume``."""
        return _block(self.aresume(), "session.aresume")

    async def aresume(self) -> Outcome:
        """Continue the last turn in place; its outcome."""
        return await self.stream(resume=True).wait()

    def wake(self) -> Outcome:
        """Run a woken turn: one with no prompt, which opens with what is
        waiting to be delivered (answers the ``sessions`` helper has
        landed, notes queued in ``inbox``); its outcome. From a
        coroutine, ``await awake``."""
        return _block(self.awake(), "session.awake")

    async def awake(self) -> Outcome:
        """:meth:`wake`, from a coroutine."""
        return await self.stream(wake=True).wait()

    def stream(
        self, prompt: str | None = None, *, resume: bool = False, wake: bool = False
    ) -> RunStream:
        """Start a turn when first iterated (or waited on), and hand back
        its events as they happen: a new turn on ``prompt``, with
        ``resume`` the last turn continued in place, or with ``wake`` a
        woken turn. When it ends, ``last`` holds the outcome."""
        if (prompt is not None) + resume + wake != 1:
            raise ValueError("a turn takes a prompt, resumes the last one, or is woken")
        return RunStream(self, prompt, resume=resume, wake=wake)

    def cancel(self) -> bool:
        """Stop the turn that is running; whether one was.

        The turn stops at the next point it can (between model calls and
        tool calls, or by interrupting the one in flight), and ends
        ``cancelled``: its work so far is kept, with a closing note. A
        tool already running on its worker thread finishes there. Safe
        to call from any thread.
        """
        running = [task for task in self._in_flight if not task.done()]
        if not running:
            return False
        self._cancel_requested = True
        try:
            here: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            here = None
        for task in running:
            loop = task.get_loop()
            if here is loop:
                self._interrupt(task)
            else:
                loop.call_soon_threadsafe(self._interrupt, task)
        return True

    def _interrupt(self, task: asyncio.Task[Outcome]) -> None:
        """On the turn's own loop: interrupt what the turn is waiting on.
        A turn not yet live, or cancelling itself (from a tool, or a
        scripted clock), stops at its next check instead."""
        if self._live and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _check_cancel(self) -> None:
        if self._cancel_requested:
            raise asyncio.CancelledError()

    def _last_interrupted(self) -> Run:
        runs = self.runs
        if not runs or runs[-1].status != "interrupted":
            ended = f"ended {runs[-1].status}" if runs else "never ran"
            raise ValueError(
                f"there is no interrupted turn to resume: the last turn {ended}"
            )
        return runs[-1]

    async def _run(
        self,
        prompt: str | None,
        emit: Callable[[TurnEvent], None],
        *,
        resume: bool = False,
        wake: bool = False,
    ) -> Outcome:
        """One turn, start to end: every event goes to ``emit`` as it
        happens, and the outcome comes back."""
        events: list[TurnEvent] = []

        def push(event: TurnEvent) -> None:
            events.append(event)
            emit(event)

        if resume:
            prior = self._last_interrupted()
            run_id = prior.run_id
            started = prior.started_at or time.time()
            messages = list(prior.messages)
            summaries = prior.compaction
            earlier = [run.messages for run in self.runs[:-1]]
        else:
            run_id = new_id()
            started = time.time()
            # a woken turn's message is what is waiting, read once the
            # turn is open
            messages = (
                []
                if wake
                else [
                    Message(id=new_id(), role="user", parts=(Text(text=prompt or ""),))
                ]
            )
            summaries = None
            earlier = [run.messages for run in self.runs]
        system = self._system()
        status: RunStatus = "completed"
        message: str | None = None
        text = ""
        external: BaseException | None = None
        calls: list[ToolCall] = []
        results: list[ToolResult] = []
        # the ToolEnded a call still owes the stream: UNANSWERED until the
        # tool returns, its own output once it has
        unended: ToolEnded | None = None
        in_request = False
        helper = self.sessions
        turn = self.ws.turn(
            run_id,
            resume=resume,
            inbox=self.inbox,
            harness=HARNESS,
            sources=(
                [lambda inbox: answer_notes(helper, inbox)]
                if helper is not None
                else []
            ),
        )
        self._live = True
        # the turn runs as a task of its own, so this is the turn's alone
        _TURN_LOOP.set(asyncio.get_running_loop())
        try:
            push(RunStarted(run_id=run_id))
            if wake:
                opening = await turn.aopening()
                messages.append(
                    Message(
                        id=new_id(), role="user", parts=(Text(text=opening or WOKEN),)
                    )
                )
            # every model call counts against max_steps, summaries too
            made = 0
            while made < self.agent.max_steps:
                self._check_cancel()
                reply: Message | None = None
                in_request = True
                prepared = await compaction.request(
                    self.ws,
                    self.agent.compaction,
                    self.agent.provider,
                    self._specs,
                    self.agent.settings,
                    system,
                    earlier,
                    messages,
                    push,
                    spare=self.agent.max_steps - made - 1,
                )
                made += prepared.calls + 1
                if prepared.usage is not None:
                    summaries = (
                        prepared.usage
                        if summaries is None
                        else summaries + prepared.usage
                    )
                fold = prepared.fold
                async for event in self.agent.provider.stream(
                    prepared.messages, self._specs, self.agent.settings
                ):
                    self._check_cancel()
                    if isinstance(event, Reply):
                        reply = event.message
                    else:
                        push(event)
                in_request = False
                if reply is None:
                    raise RuntimeError("the provider's stream ended without a reply")
                if fold is not None:
                    reply = replace(reply, fold=fold)
                messages.append(reply)
                if reply.usage is not None:
                    push(
                        Usage(
                            input_tokens=reply.usage.input_tokens,
                            cached_tokens=reply.usage.cache_read_tokens,
                        )
                    )
                calls = [p for p in reply.parts if isinstance(p, ToolCall)]
                if not calls:
                    text = reply.text
                    after = self._on_stop(reply)
                    if isinstance(after, Message):
                        messages.append(after)
                        continue
                    if after is not None:
                        status, message = "failed", after
                    break
                results = []
                done = False
                for call in calls:
                    self._check_cancel()
                    push(
                        ToolStarted(
                            call_id=call.call_id, name=call.name, args=dict(call.args)
                        )
                    )
                    unended = ToolEnded(
                        call_id=call.call_id,
                        name=call.name,
                        result=CUT_OFF,
                        is_error=True,
                    )
                    output, is_error = self._after_call(*await self._call(call))
                    # the tool ran, and may have changed the workspace: its
                    # result stands from here, however the turn ends
                    unended = ToolEnded(
                        call_id=call.call_id,
                        name=call.name,
                        result=output,
                        is_error=is_error,
                    )
                    results.append(
                        ToolResult(
                            call_id=call.call_id,
                            name=call.name,
                            content=output,
                            is_error=is_error,
                        )
                    )
                    delivered, notes = await turn.adeliver(output)
                    if notes:
                        push(_delivered(notes))
                        results[-1] = replace(results[-1], content=delivered)
                    push(unended)
                    unended = None
                    done = self._done()
                    if done:
                        break
                if done:
                    # the calls left in ``calls`` get their results as the
                    # run ends, as a cut-short run's do
                    break
                messages.append(Message(id=new_id(), role="tool", parts=tuple(results)))
                calls = []
            else:
                status = "failed"
                message = (
                    f"the model made {self.agent.max_steps} calls without "
                    "finishing the turn"
                )
        except asyncio.CancelledError as exc:
            status = "cancelled"
            if self._cancel_requested:
                message = STOPPED
                # a stop asked for is handled here, so the task ends with
                # the outcome rather than cancelled (3.11+ counts requests)
                uncancel = getattr(asyncio.current_task(), "uncancel", None)
                if uncancel is not None:
                    uncancel()
            else:
                message = "cancelled"
                external = exc
        except Exception as exc:  # noqa: BLE001 - the outcome says how it ended
            transient = in_request and self.agent.provider.transient(exc)
            status = "interrupted" if transient else "failed"
            message = _describe(exc)
            if not in_request:
                _logger.warning("turn %s failed", run_id, exc_info=True)
        except BaseException as exc:
            status, message, external = "cancelled", _describe(exc), exc
        finally:
            self._live = False
            if turn.open:
                if unended is not None:
                    push(unended)
                if calls:
                    # every tool call needs its result, or the next request
                    # is refused: the ones the turn never answered say so
                    answered = {r.call_id for r in results}
                    cut_off = unended.call_id if unended is not None else None
                    unanswered = [
                        ToolResult(
                            call_id=c.call_id,
                            name=c.name,
                            content=CUT_OFF if c.call_id == cut_off else UNANSWERED,
                            is_error=True,
                        )
                        for c in calls
                        if c.call_id not in answered
                    ]
                    messages.append(
                        Message(id=new_id(), role="tool", parts=(*results, *unanswered))
                    )
                if status in ("cancelled", "failed"):
                    messages.append(closing_note(message or status))
                run = Run(
                    run_id=run_id,
                    status=status,
                    messages=tuple(messages),
                    started_at=started,
                    ended_at=time.time(),
                    compaction=summaries,
                )
                turn.end(status, body=dump_run(run), message=message)
        push(RunEnded(status=status, message=message))
        self.last = Outcome(
            status=status,
            text=text,
            run_id=run_id,
            head=self.ws.head,
            events=tuple(events),
            message=message,
        )
        if external is not None:
            raise external
        return self.last

    # -- the loop's parts -----------------------------------------------------------

    async def _call(self, call: ToolCall) -> tuple[str, bool]:
        """The tool's text and whether it failed. A tool the model made
        up, arguments that do not fit, or a tool that raises is a failed
        call the model reads, not an error out of the turn."""
        tool = self._tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._tools))
            return f"there is no tool named {call.name!r}; the tools are {known}", True
        try:
            output = await tool.acall(**call.args)
        except Exception as exc:  # noqa: BLE001 - the model reads it
            return f"{type(exc).__name__}: {exc}", True
        return output.text, output.is_error


_END: Any = object()


class RunStream:
    """One turn's events, as they happen.

    The turn is a task of its own, started by the first iteration or by
    :meth:`wait`. Iterating watches it: leaving early stops watching,
    and the turn goes on to its end, landing its commit as usual. An
    error that ended the turn is raised to whoever iterates or waits.
    """

    def __init__(
        self,
        session: Session,
        prompt: str | None,
        *,
        resume: bool = False,
        wake: bool = False,
    ) -> None:
        self._session = session
        self._prompt = prompt
        self._resume = resume
        self._wake = wake
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task: asyncio.Task[Outcome] | None = None

    def _start(self) -> asyncio.Task[Outcome]:
        if self._task is None:
            session = self._session
            loop = asyncio.get_running_loop()
            session._bind(loop)
            session._cancel_requested = False
            task = loop.create_task(
                session._run(
                    self._prompt,
                    self._queue.put_nowait,
                    resume=self._resume,
                    wake=self._wake,
                )
            )
            # the end of the stream, however the task ends: even cancelled
            # before it ran a line
            task.add_done_callback(lambda _: self._queue.put_nowait(_END))
            self._task = session._track(task)
        return self._task

    @property
    def outcome(self) -> Outcome | None:
        """The turn's outcome, once it has ended."""
        task = self._task
        if task is None or not task.done() or task.cancelled():
            return None
        return task.result() if task.exception() is None else None

    def __aiter__(self) -> RunStream:
        return self

    async def __anext__(self) -> TurnEvent:
        task = self._start()
        item = await self._queue.get()
        if item is _END:
            self._queue.put_nowait(_END)  # every later call ends too
            error = None if task.cancelled() else task.exception()
            if error is not None:
                raise error
            raise StopAsyncIteration
        return item

    async def wait(self) -> Outcome:
        """Run the turn to its end, if it is not running already, and
        return its outcome. A waiter that is cancelled leaves the turn
        running."""
        return await asyncio.shield(self._start())
