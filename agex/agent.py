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
import time
from collections.abc import AsyncIterator
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

__all__ = ["HARNESS", "Agent", "Outcome", "Session"]

HARNESS = "agex"
"""The name agex's conversations are stored under, in the workspace's
conversation index."""

MAX_STEPS = 100
"""How many model calls a run may make before it is stopped."""


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
    same from a coroutine; ``stream`` yields the turn's events as they
    happen (``nontainer.turns``), and leaves the outcome on ``last``.
    ``inbox`` takes notes mid-turn: each lands on the next tool result.

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

    def __repr__(self) -> str:
        return f"<Session {self.ws.session!r} of {self.agent!r}>"

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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asay(prompt))
        raise RuntimeError(
            "say() blocks, and this thread is running an event loop; "
            "use `await session.asay(...)` there"
        )

    async def asay(self, prompt: str) -> Outcome:
        """Run one turn; its outcome."""
        async for _ in self.stream(prompt):
            pass
        assert self.last is not None
        return self.last

    async def stream(self, prompt: str) -> AsyncIterator[TurnEvent]:
        """Run one turn, yielding its events as they happen. When the
        stream ends, ``last`` holds the outcome. Leaving the stream
        before its end cancels the run."""
        run_id = new_id()
        events: list[TurnEvent] = []
        history = self._history()
        system = self._system()
        messages: list[Message] = [
            Message(id=new_id(), role="user", parts=(Text(text=prompt),))
        ]
        started = time.time()
        status: RunStatus = "completed"
        message: str | None = None
        text = ""
        turn = self.ws.turn(run_id, inbox=self.inbox, harness=HARNESS)
        try:
            opened = RunStarted(run_id=run_id)
            events.append(opened)
            yield opened
            for _ in range(self.agent.max_steps):
                reply: Message | None = None
                async for event in self.agent.provider.stream(
                    [system, *history, *messages],
                    self._specs,
                    self.agent.settings,
                ):
                    if isinstance(event, Reply):
                        reply = event.message
                        continue
                    events.append(event)
                    yield event
                if reply is None:
                    raise RuntimeError("the provider's stream ended without a reply")
                messages.append(reply)
                if reply.usage is not None:
                    usage = Usage(
                        input_tokens=reply.usage.input_tokens,
                        cached_tokens=reply.usage.cache_read_tokens,
                    )
                    events.append(usage)
                    yield usage
                calls = [p for p in reply.parts if isinstance(p, ToolCall)]
                if not calls:
                    text = reply.text
                    break
                results: list[ToolResult] = []
                for call in calls:
                    async for event in self._run_tool(call, turn, results):
                        events.append(event)
                        yield event
                messages.append(Message(id=new_id(), role="tool", parts=tuple(results)))
            else:
                status = "failed"
                message = (
                    f"the model made {self.agent.max_steps} calls without "
                    "finishing the turn"
                )
        except BaseException as exc:
            status = "failed" if isinstance(exc, Exception) else "cancelled"
            message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            raise
        finally:
            if turn.open:
                run = Run(
                    run_id=run_id,
                    status=status,
                    messages=tuple(messages),
                    started_at=started,
                    ended_at=time.time(),
                )
                turn.end(status, body=dump_run(run), message=message)
        ended = RunEnded(status=status, message=message)
        events.append(ended)
        self.last = Outcome(
            status=status,
            text=text,
            run_id=run_id,
            head=self.ws.head,
            events=tuple(events),
            message=message,
        )
        yield ended

    # -- the loop's parts -----------------------------------------------------------

    async def _run_tool(
        self, call: ToolCall, turn: Any, results: list[ToolResult]
    ) -> AsyncIterator[TurnEvent]:
        """Run one tool call: its start, the notes its result delivers,
        its end. The result, notes included, goes into ``results`` for
        the model."""
        yield ToolStarted(call_id=call.call_id, name=call.name, args=dict(call.args))
        output, is_error = await self._call(call)
        delivered, notes = await turn.adeliver(output)
        if notes:
            yield Delivered(
                notes=tuple(
                    DeliveredNote(
                        id=n.id, text=n.text, kind=n.kind, label=n.label, job=n.job
                    )
                    for n in notes
                )
            )
        yield ToolEnded(
            call_id=call.call_id, name=call.name, result=output, is_error=is_error
        )
        results.append(
            ToolResult(
                call_id=call.call_id,
                name=call.name,
                content=delivered,
                is_error=is_error,
            )
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
