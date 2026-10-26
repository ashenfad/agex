"""agex's stored runs, read into nontainer's neutral trace shape.

:class:`AgexTraces` is the :class:`~nontainer.traces.TraceSource` for
the conversations agex writes: each run under
``__conversation__/runs/<id>`` is a :class:`~agex.record.Run` as
:func:`~agex.record.dump_run` stored it. Register it under agex's
harness name and nontainer renders an agex session as files, at
``.trace`` beside a worktree's files (``ws-git worktree add --trace``)::

    from nontainer import Store
    from agex.traces import AgexTraces

    store = Store(path, traces={"agex": AgexTraces()})

What it reads from a run:

- **status**: already nontainer's words; a run with none is ``running``.
- **times**: ``started_at`` and ``ended_at``.
- **model**: the models its replies name, in the order they first
  replied, joined by ``", "``.
- **tokens**: :attr:`Run.usage <agex.record.Run.usage>`, its replies'
  and its summary calls'. agex counts no reasoning tokens, so that
  count is unknown.
- **events**, from the messages in order: a ``user`` message's text; an
  ``assistant`` message's parts as they came, its reasoning, its text,
  and one tool event per call, whose result is the ``tool`` message
  part with the call's id among the ones that follow it; and a result
  no call claims, on its own. A result's images are left out.

agex builds its system prompt for each request and doesn't store it,
so a trace has no system prompt unless a run holds a ``system``
message.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nontainer.traces import TraceEvent, TraceRun, TraceTokens

from .record import Message, Run, Text, Thinking, ToolCall, ToolResult, load_run

__all__ = ["AgexTraces"]


class AgexTraces:
    """The trace source for runs agex stored. Register it under the
    harness name agex's index carries:
    ``Store(..., traces={"agex": AgexTraces()})``."""

    def read(self, body: Any) -> TraceRun:
        if not isinstance(body, Mapping):
            raise ValueError(f"an agex run is an object, got {type(body).__name__}")
        run = load_run(body)
        usage = run.usage
        return TraceRun(
            id=run.run_id,
            status=run.status or "running",
            started=run.started_at,
            ended=run.ended_at,
            model=_models(run),
            tokens=TraceTokens(
                input=usage.input_tokens,
                output=usage.output_tokens,
                cache_read=usage.cache_read_tokens,
                cache_write=usage.cache_write_tokens,
            ),
            system=next(
                (m.text for m in run.messages if m.role == "system" and m.text), None
            ),
            events=tuple(_events(run.messages)),
        )


def _models(run: Run) -> str | None:
    models: list[str] = []
    for message in run.messages:
        if (
            message.role == "assistant"
            and message.model
            and message.model not in models
        ):
            models.append(message.model)
    return ", ".join(models) or None


def _tool_event(call: ToolCall | None, result: ToolResult | None) -> TraceEvent:
    if call is not None:
        name, args = call.name, dict(call.args)
    else:
        assert result is not None
        name, args = result.name, None
    return TraceEvent(
        "tool",
        None if result is None else result.content,
        tool=name,
        args=args,
        error=result is not None and result.is_error,
    )


def _events(messages: tuple[Message, ...]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        i += 1
        if message.role == "user":
            if message.text:
                events.append(TraceEvent("user", message.text))
        elif message.role == "assistant":
            # the results of this reply's calls: the tool messages right after it
            unclaimed: list[ToolResult] = []
            while i < len(messages) and messages[i].role == "tool":
                unclaimed.extend(
                    p for p in messages[i].parts if isinstance(p, ToolResult)
                )
                i += 1
            texts: list[str] = []

            def said() -> None:
                if "".join(texts):
                    events.append(TraceEvent("assistant", "".join(texts)))
                texts.clear()

            for part in message.parts:
                if isinstance(part, Text):
                    texts.append(part.text)
                    continue
                said()
                if isinstance(part, Thinking):
                    if part.text:
                        events.append(TraceEvent("thinking", part.text))
                elif isinstance(part, ToolCall):
                    result = next(
                        (r for r in unclaimed if r.call_id == part.call_id), None
                    )
                    if result is not None:
                        unclaimed.remove(result)
                    events.append(_tool_event(part, result))
            said()
            events.extend(_tool_event(None, r) for r in unclaimed)
        elif message.role == "tool":
            events.extend(
                _tool_event(None, p) for p in message.parts if isinstance(p, ToolResult)
            )
    return events
