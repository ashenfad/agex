"""Compaction: agex folds its own conversations, within a run as well.

nontainer owns the record of a fold and its rules (``nontainer.
compaction``, ``docs/compaction.md`` there): a :class:`~nontainer.
compaction.Fold` in the workspace's ``__compaction__/`` plane, in force
while the message it ends at is in the history, taken back by a rewind,
carried by a full fork, and never in a stored run. How a loop folds is
its own, and this is agex's:

- **What the model is sent.** The fold in force replaces every message
  up to its anchor with the summary pair (``summary_message`` and
  ``ACK``). When the anchor is inside a run, that run's opening message
  (the person's prompt, or a task's brief) is sent again after the
  pair, so the model always has the request it is working on, and the
  roles still alternate. Everything after the anchor is sent as it is.
- **When it folds.** Before each model call, when the agent has a
  :class:`~nontainer.compaction.Policy` and the request would reach its
  budget. A request is measured from the provider's report for the
  latest reply, adjusted for any change of fold since (each reply says
  which fold its request had, ``Message.fold``), plus an estimate for
  what came after it.
- **What a fold covers.** Everything before the latest step of the run
  in progress: the earlier runs, and the run's own earlier steps, so a
  long task folds within its run. A fold ends at a tool result or at the
  end of an earlier run, never between a tool call and its result.
- **Who writes the summary.** The agent's own model, sent the request
  it was about to be sent up to the fold's end, the same tools, and the
  instruction to summarise: almost all of that is in the provider's
  cache. A reply with no text, or a request that fails, falls back to a
  transcript with tool output cut down, in chunks when even that is too
  large, sent without tools.

The budget should sit well above what no fold can take out (the system
prompt, the tools, the opening message): a request still over it after
a fold folds again at the next step, a summary call each time.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from nontainer import Workspace
from nontainer.compaction import (
    ACK,
    MARK,
    Fold,
    Item,
    Policy,
    chunks,
    estimate_tokens,
    folds,
    in_force,
    record,
    reduce,
    summary_message,
    summary_request,
    transcript,
)
from nontainer.turns import Compacted, TurnEvent

from .providers import Provider, Reply, Settings, ToolSpec
from .record import Message, Text, Thinking, ToolCall, ToolResult, new_id

__all__ = ["request"]

_SUMMARY_ID = MARK + "summary:"
_ACK_ID = MARK + "ack:"


@dataclass(frozen=True)
class _Seq:
    """The conversation as one sequence: the earlier runs' messages,
    then the run in progress's, each with the index of its run."""

    messages: list[Message]
    runs: list[int]
    current: int

    @classmethod
    def of(
        cls, earlier: Sequence[Sequence[Message]], current: Sequence[Message]
    ) -> _Seq:
        messages: list[Message] = []
        runs: list[int] = []
        for n, run in enumerate(earlier):
            messages.extend(run)
            runs.extend([n] * len(run))
        messages.extend(current)
        runs.extend([len(earlier)] * len(current))
        return cls(messages, runs, len(earlier))

    def index(self, message_id: str | None) -> int | None:
        if message_id is None:
            return None
        return next(
            (i for i, m in enumerate(self.messages) if m.id == message_id), None
        )

    def removed(self, anchor: str | None) -> set[int]:
        """What a fold ending at ``anchor`` keeps out of a request: every
        message up to it, but the opening message of a run it ends
        inside."""
        end = self.index(anchor)
        if end is None:
            return set()
        run = [i for i, r in enumerate(self.runs) if r == self.runs[end]]
        kept = run[0] if end != run[-1] else None
        return {i for i in range(end + 1) if i != kept}

    def cut(self) -> int | None:
        """Where a new fold would end: the latest tool result before the
        run's latest step, else the end of the earlier runs."""
        current = [i for i, r in enumerate(self.runs) if r == self.current]
        replies = [i for i in current if self.messages[i].role == "assistant"]
        latest = replies[-1] if replies else None
        steps = [
            i
            for i in current[1:]
            if self.messages[i].role == "tool" and (latest is None or i < latest)
        ]
        if steps:
            return steps[-1]
        earlier = [i for i, r in enumerate(self.runs) if r != self.current]
        return earlier[-1] if earlier else None


def _pair(fold: Fold) -> list[Message]:
    return [
        Message(
            id=_SUMMARY_ID + fold.through,
            role="user",
            parts=(Text(text=summary_message(fold.summary)),),
        ),
        Message(id=_ACK_ID + fold.through, role="assistant", parts=(Text(text=ACK),)),
    ]


def _view(seq: _Seq, fold: Fold | None, upto: int | None = None) -> list[Message]:
    """What the model is sent of the conversation with ``fold`` in force,
    up to message ``upto`` when given."""
    removed = seq.removed(fold.through if fold else None)
    kept = [
        m
        for i, m in enumerate(seq.messages)
        if i not in removed and (upto is None or i <= upto)
    ]
    return (_pair(fold) if fold else []) + kept


def _text(message: Message) -> str:
    parts = []
    for part in message.parts:
        if isinstance(part, (Text, Thinking)):
            parts.append(part.text)
        elif isinstance(part, ToolCall):
            parts.append(
                json.dumps({"name": part.name, "args": part.args}, default=str)
            )
        elif isinstance(part, ToolResult):
            parts.append(part.content)
    return "\n".join(parts)


def _size(messages: Sequence[Message]) -> int:
    return sum(estimate_tokens(_text(m)) + 4 for m in messages)


def _measure(
    seq: _Seq,
    fold: Fold | None,
    recorded: Sequence[Fold],
    system: Message,
    specs: Sequence[ToolSpec],
) -> int:
    """The size, in tokens, of the request ``fold`` gives: the
    provider's report for the latest reply, adjusted for the fold its
    request had and this one has, plus an estimate for the messages
    from that reply on; all estimated when no reply reported one."""
    latest = next(
        (
            i
            for i in range(len(seq.messages) - 1, -1, -1)
            if seq.messages[i].role == "assistant"
            and seq.messages[i].usage is not None
            and seq.messages[i].usage.input_tokens  # type: ignore[union-attr]
        ),
        None,
    )
    if latest is None:
        tools = json.dumps([s.parameters for s in specs], default=str)
        return _size([system, *_view(seq, fold)]) + estimate_tokens(tools)
    reply = seq.messages[latest]
    assert reply.usage is not None
    then = next((f for f in reversed(recorded) if f.through == reply.fold), None)
    gone_then = {i for i in seq.removed(reply.fold) if i < latest}
    gone_now = seq.removed(fold.through if fold else None)
    msgs = seq.messages
    size = reply.usage.input_tokens + _size(msgs[latest:])
    size -= _size([msgs[i] for i in sorted(gone_now - gone_then)])
    size += _size([msgs[i] for i in sorted(gone_then - gone_now)])
    size += _size(_pair(fold) if fold else []) - _size(_pair(then) if then else [])
    return max(size, 0)


def _items(messages: Sequence[Message]) -> list[Item]:
    items = []
    for m in messages:
        for part in m.parts:
            if isinstance(part, Text) and part.text:
                items.append(Item(m.role, part.text))
            elif isinstance(part, ToolCall):
                text = json.dumps({"name": part.name, "args": part.args}, default=str)
                items.append(Item(m.role, text, "tool_call"))
            elif isinstance(part, ToolResult):
                items.append(Item(m.role, part.content, "tool_result"))
    return items


async def _reply(
    provider: Provider,
    messages: Sequence[Message],
    specs: Sequence[ToolSpec],
    settings: Settings,
) -> str:
    """The text of the model's reply to ``messages``."""
    async for event in provider.stream(messages, specs, settings):
        if isinstance(event, Reply):
            return event.message.text.strip()
    return ""


def _user(text: str) -> Message:
    return Message(id=new_id(), role="user", parts=(Text(text=text),))


async def _summarize(
    provider: Provider,
    policy: Policy,
    settings: Settings,
    specs: Sequence[ToolSpec],
    system: Message,
    prefix: list[Message],
    size: int,
) -> str:
    """The summary of ``prefix``, or ``""`` when none could be written."""
    ask = _user(summary_request())
    if policy.fits(size + estimate_tokens(summary_request())):
        try:
            text = await _reply(provider, [system, *prefix, ask], specs, settings)
        except Exception:  # noqa: BLE001 - the reduced path is the answer
            text = ""
        if text:
            return text
    room = (policy.window or policy.budget) // 2
    parts = chunks(reduce(_items(prefix)), room)
    summary: str | None = None
    for n, part in enumerate(parts, 1):
        label = None if len(parts) == 1 else f"part {n} of {len(parts)}"
        text = summary_request(transcript(part), prior=summary, part=label)
        try:
            summary = await _reply(provider, [_user(text)], (), settings) or summary
        except Exception:  # noqa: BLE001 - no fold this time
            return ""
    return summary or ""


async def request(
    ws: Workspace,
    policy: Policy | None,
    provider: Provider,
    specs: Sequence[ToolSpec],
    settings: Settings,
    system: Message,
    earlier: Sequence[Sequence[Message]],
    current: Sequence[Message],
    emit: Callable[[TurnEvent], None],
) -> tuple[list[Message], str | None]:
    """The request for the next model call, and the anchor of the fold
    in force in it (``None``: none is), which the reply carries as its
    ``fold``. Over the budget, it folds first: the fold is recorded in
    ``ws`` (landing with the turn's commit) and emitted as
    ``Compacted``."""
    seq = _Seq.of(earlier, current)
    recorded = folds(ws)
    fold = in_force(recorded, [m.id for m in seq.messages])
    if policy is not None:
        size = _measure(seq, fold, recorded, system, specs)
        end = seq.cut()
        start = seq.index(fold.through) if fold else None
        if policy.due(size) and end is not None and (start is None or end > start):
            prefix = _view(seq, fold, upto=end)
            summary = await _summarize(
                provider, policy, settings, specs, system, prefix, size
            )
            if summary:
                gone = seq.removed(seq.messages[end].id)
                new = Fold(
                    through=seq.messages[end].id,
                    summary=summary,
                    runs=len({seq.runs[i] for i in gone}),
                    tokens_before=size,
                    model=provider.name,
                )
                new = Fold(
                    **{
                        **new.to_dict(),
                        "tokens_after": _measure(
                            seq, new, [*recorded, new], system, specs
                        ),
                    }
                )
                record(ws, new)
                emit(Compacted(through=new.through, runs=new.runs))
                fold = new
    view = _view(seq, fold)
    return [system, *view], fold.through if fold else None
