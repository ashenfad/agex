"""The stored run record: what agex keeps of a run, in its own format.

A run is the messages one turn exchanged with the model, with how it
ended. It is stored in the workspace's conversation plane (one run body
per run id, see ``nontainer.conversation``), so its format is agex's
and stays agex's: providers' message types are mapped to and from it at
the edge, never stored. The record is also what agex in other
languages mirrors, so it is plain frozen dataclasses with a ``kind`` on
every union member, read and written as JSON by :func:`dump_run` and
:func:`load_run`.

Every message has a stable ``id``. Compaction anchors a fold on a
message id, so an id never changes once the message is stored.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from nontainer.conformance.codec import dump, load
from nontainer.turns import RunStatus

__all__ = [
    "FORMAT",
    "Message",
    "Part",
    "Role",
    "Run",
    "Text",
    "Thinking",
    "ToolCall",
    "ToolResult",
    "Usage",
    "dump_run",
    "load_run",
    "new_id",
]

FORMAT = 1
"""The record's format version. Bumped only when a stored field changes
meaning or goes away; an added field with a default does not bump it."""


def new_id() -> str:
    """A fresh message or run id."""
    return uuid.uuid4().hex


# -- parts ----------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Text:
    """Prose: what a person said, or what the model replied. A model's
    text keeps the provider's own ``id`` for it and any ``details`` the
    provider attached, which some send back."""

    kind: Literal["text"] = "text"
    text: str
    id: str | None = None
    provider: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class Thinking:
    """The model's reasoning, kept so it can be sent back.

    Some providers only accept reasoning back with the ``signature``
    they issued for it, and some carry more in ``details`` (encrypted
    content, item ids). ``provider`` names who wrote it: reasoning from
    one provider is not sent to another.
    """

    kind: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None
    id: str | None = None
    provider: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class ToolCall:
    """A tool call the model made. ``call_id`` is the provider's id for
    it, which the matching :class:`ToolResult` carries back.

    ``id`` is the provider's id for the call as an item of its reply (an
    OpenAI Responses ``fc_...`` id), and ``details`` what else it
    attached (Gemini's ``thought_signature``). Both go back with the
    call, or the provider may refuse the next request.
    """

    kind: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    provider: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class ToolResult:
    """What a tool call returned, to the model. ``is_error`` says the
    call failed and ``content`` says how."""

    kind: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    content: str
    is_error: bool = False


Part = Text | Thinking | ToolCall | ToolResult

Role = Literal["system", "user", "assistant", "tool"]
"""Who a message is from. ``tool`` messages hold tool results; an
``assistant`` message is one model reply."""


# -- messages and runs ------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Usage:
    """What a model call cost, in tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, kw_only=True)
class Message:
    """One message of a run.

    An ``assistant`` message is one model reply, and says which
    ``model`` (and ``provider``) wrote it, what it cost (``usage``) and
    the anchor of the compaction fold in force in the request it
    answers (``fold``), which is how a later request is measured from
    its usage; those are ``None`` on the other roles.
    """

    id: str
    role: Role
    parts: tuple[Part, ...] = ()
    model: str | None = None
    provider: str | None = None
    usage: Usage | None = None
    fold: str | None = None

    @property
    def text(self) -> str:
        """The message's prose, its text parts joined."""
        return "".join(p.text for p in self.parts if isinstance(p, Text))


@dataclass(frozen=True, kw_only=True)
class Run:
    """One run: the messages a turn exchanged with the model, in order,
    and how it ended (``None`` while it runs)."""

    format: int = FORMAT
    run_id: str
    status: RunStatus | None = None
    messages: tuple[Message, ...] = ()
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def usage(self) -> Usage:
        """What the run's model calls cost, summed."""
        total = Usage()
        for message in self.messages:
            if message.usage is not None:
                total = total + message.usage
        return total


def dump_run(run: Run) -> dict[str, Any]:
    """``run`` as JSON data, the stored form."""
    return dump(run)


def load_run(data: Mapping[str, Any]) -> Run:
    """A run read back from :func:`dump_run`'s output. Raises
    ``ValueError`` for data that does not fit the format, including a
    format newer than this agex knows."""
    version = data.get("format", FORMAT)
    if not isinstance(version, int) or version > FORMAT:
        raise ValueError(f"run record format {version!r} is newer than {FORMAT}")
    return load(Run, data)
