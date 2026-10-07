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
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nontainer import NotSupportedError, Profile, Workspace, conversation
from nontainer.adapters.render import toolkit_instructions
from nontainer.adapters.tools import Tool, Toolset
from nontainer.inbox import Inbox
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

__all__ = [
    "CLOSING_NOTE",
    "HARNESS",
    "Agent",
    "Outcome",
    "RunStream",
    "Session",
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

UNANSWERED = "not run: the turn ended before this call returned"
"""The result a tool call gets when its turn ended first."""


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


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _block(coro: Any, name: str) -> Outcome:
    """Run ``coro`` to its end on a loop of its own, from code that has
    none; refused where a loop is running, which this would block."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "this blocks, and this thread is running an event loop; "
        f"use `await session.{name}(...)` there"
    )


@dataclass(frozen=True)
class Outcome:
    """How a run went.

    ``status`` is how it ended, ``text`` the model's last reply, and
    ``head`` the commit the session stands at afterwards (``None`` on a
    workspace that does not version). ``events`` are what the run
    streamed, and ``message`` says why it did not complete, when it
    did not.
    """

    status: RunStatus
    text: str
    run_id: str
    head: str | None
    events: tuple[TurnEvent, ...] = ()
    message: str | None = None


class Agent:
    """A loop over a model.

    ``model`` is a :class:`~agex.providers.Provider` or a model name
    pydantic-ai knows (``"anthropic:claude-sonnet-5-5"``). ``primer``
    opens every request, ahead of the workspace's own tool instructions.
    ``profile`` is the environment for worlds the agent creates itself; a
    session uses its workspace's own. ``max_steps`` bounds the model
    calls in one run.
    """

    def __init__(
        self,
        model: Provider | str,
        *,
        primer: str = "",
        profile: Profile | None = None,
        settings: Settings = Settings(),
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.provider: Provider = (
            PydanticAIProvider(model) if isinstance(model, str) else model
        )
        self.primer = primer
        self.profile = profile
        self.settings = settings
        self.max_steps = max_steps

    def __repr__(self) -> str:
        return f"<Agent {self.provider.name}>"

    def session(self, ws: Workspace, *, inbox: Inbox | None = None) -> Session:
        """A session driving ``ws``, which may already hold an agex
        conversation (it continues) or none (it starts one)."""
        return Session(self, ws, inbox=inbox)


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

    A turn ends ``completed``, ``cancelled``, ``failed`` (an error, or
    ``max_steps`` model calls) or ``interrupted`` (a provider error worth
    retrying, see :meth:`~agex.providers.Provider.transient`), which
    ``resume`` continues in place. A cancelled or failed turn keeps its
    work with a closing note, so the model remembers what it did; none
    of the endings raise.

    A workspace whose conversation another harness wrote is refused: the
    runs it holds are in that harness's format.
    """

    def __init__(self, agent: Agent, ws: Workspace, *, inbox: Inbox | None = None):
        index = conversation.index_of(ws)
        if index is not None and index.harness != HARNESS:
            raise NotSupportedError(
                f"session {ws.session!r} holds a conversation {index.harness!r} "
                "wrote; agex reads only its own"
            )
        self.agent = agent
        self.ws = ws
        self.inbox = inbox if inbox is not None else Inbox()
        self.toolset = Toolset(ws, vision=False)
        self._tools: dict[str, Tool] = {t.name: t for t in self.toolset.tools()}
        self._specs = [ToolSpec.of(t) for t in self._tools.values()]
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

    def _history(self) -> list[Message]:
        return [message for run in self.runs for message in run.messages]

    def _system(self) -> Message:
        instructions = toolkit_instructions(
            self.ws,
            split=self.toolset.split,
            turn_commits=not self.ws.autocommit,
        )
        text = "\n\n".join(t for t in (self.agent.primer, instructions) if t)
        return Message(id="system", role="system", parts=(Text(text=text),))

    # -- the front doors ------------------------------------------------------------

    def say(self, prompt: str) -> Outcome:
        """Run one turn; its outcome. From a coroutine, ``await asay``."""
        return _block(self.asay(prompt), "asay")

    async def asay(self, prompt: str) -> Outcome:
        """Run one turn; its outcome."""
        return await self.stream(prompt).wait()

    def resume(self) -> Outcome:
        """Continue the last turn in place, which an interruption cut
        short; its outcome. From a coroutine, ``await aresume``."""
        return _block(self.aresume(), "aresume")

    async def aresume(self) -> Outcome:
        """Continue the last turn in place; its outcome."""
        return await self.stream(resume=True).wait()

    def stream(self, prompt: str | None = None, *, resume: bool = False) -> RunStream:
        """Start a turn when first iterated (or waited on), and hand back
        its events as they happen: a new turn on ``prompt``, or with
        ``resume`` the last turn continued in place. When it ends,
        ``last`` holds the outcome."""
        if resume == (prompt is not None):
            raise ValueError("a turn takes a prompt, or resumes the last one")
        return RunStream(self, prompt, resume=resume)

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
            history = [m for run in self.runs[:-1] for m in run.messages]
        else:
            run_id = new_id()
            started = time.time()
            messages = [
                Message(id=new_id(), role="user", parts=(Text(text=prompt or ""),))
            ]
            history = self._history()
        system = self._system()
        status: RunStatus = "completed"
        message: str | None = None
        text = ""
        external: BaseException | None = None
        calls: list[ToolCall] = []
        results: list[ToolResult] = []
        inflight: ToolCall | None = None
        in_request = False
        turn = self.ws.turn(run_id, resume=resume, inbox=self.inbox, harness=HARNESS)
        self._live = True
        try:
            push(RunStarted(run_id=run_id))
            for _ in range(self.agent.max_steps):
                self._check_cancel()
                reply: Message | None = None
                in_request = True
                async for event in self.agent.provider.stream(
                    [system, *history, *messages],
                    self._specs,
                    self.agent.settings,
                ):
                    self._check_cancel()
                    if isinstance(event, Reply):
                        reply = event.message
                    else:
                        push(event)
                in_request = False
                if reply is None:
                    raise RuntimeError("the provider's stream ended without a reply")
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
                    break
                results = []
                for call in calls:
                    self._check_cancel()
                    inflight = call
                    results.append(await self._run_tool(call, turn, push))
                    inflight = None
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
                if inflight is not None:
                    push(
                        ToolEnded(
                            call_id=inflight.call_id,
                            name=inflight.name,
                            result=UNANSWERED,
                            is_error=True,
                        )
                    )
                if calls:
                    # every tool call needs its result, or the next request
                    # is refused: the ones the turn never answered say so
                    answered = {r.call_id for r in results}
                    unanswered = [
                        ToolResult(
                            call_id=c.call_id,
                            name=c.name,
                            content=UNANSWERED,
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

    async def _run_tool(
        self, call: ToolCall, turn: Any, push: Callable[[TurnEvent], None]
    ) -> ToolResult:
        """Run one tool call, pushing its start, the notes its result
        delivers, and its end; the result, notes included, for the
        model."""
        push(ToolStarted(call_id=call.call_id, name=call.name, args=dict(call.args)))
        output, is_error = await self._call(call)
        delivered, notes = await turn.adeliver(output)
        if notes:
            push(
                Delivered(
                    notes=tuple(
                        DeliveredNote(
                            id=n.id, text=n.text, kind=n.kind, label=n.label, job=n.job
                        )
                        for n in notes
                    )
                )
            )
        push(
            ToolEnded(
                call_id=call.call_id, name=call.name, result=output, is_error=is_error
            )
        )
        return ToolResult(
            call_id=call.call_id, name=call.name, content=delivered, is_error=is_error
        )

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
        self, session: Session, prompt: str | None, *, resume: bool = False
    ) -> None:
        self._session = session
        self._prompt = prompt
        self._resume = resume
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task: asyncio.Task[Outcome] | None = None

    def _start(self) -> asyncio.Task[Outcome]:
        if self._task is None:
            session = self._session
            session._cancel_requested = False
            task = asyncio.get_running_loop().create_task(
                session._run(self._prompt, self._queue.put_nowait, resume=self._resume)
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
