"""How agex talks to a model.

A :class:`Provider` takes the conversation so far (agex's own
messages, :class:`~agex.record.Message`), the tools the model may call, and
the request's settings, and streams the reply: text and reasoning as
they arrive, then the whole reply as one assistant message, usage
included. Everything provider-specific (message formats, streaming
protocols, cache controls) stays behind it.

:class:`~agex.providers.pydanticai.PydanticAIProvider` is the first
implementation, over ``pydantic_ai.direct``, and
:class:`~agex.providers.scripted.ScriptedProvider` replies from a
script, for tests. A provider whose newest features lag in pydantic-ai
can implement the protocol over its own SDK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from nontainer.turns import TextDelta, ThinkingDelta

from ..record import Message

__all__ = ["Provider", "ProviderEvent", "Reply", "Settings", "Thinking", "ToolSpec"]


@dataclass(frozen=True, kw_only=True)
class ToolSpec:
    """A tool the model may call: its name, what it does, and the JSON
    Schema of its arguments. ``ToolSpec.of(tool)`` takes one from a
    ``nontainer.adapters.tools.Tool``."""

    name: str
    description: str
    parameters: dict[str, Any]

    @classmethod
    def of(cls, tool: Any) -> ToolSpec:
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=dict(tool.parameters),
        )


Thinking = Literal["minimal", "low", "medium", "high", "xhigh"]
"""How hard a model reasons before it answers, where it can."""


@dataclass(frozen=True, kw_only=True)
class Settings:
    """A request's settings.

    ``thinking`` asks for reasoning (``True``, or an effort) on any
    provider that offers it, each in its own terms; ``False`` asks for
    none, and ``None`` leaves the provider's default. ``extra`` passes
    provider-specific settings through as they are (cache controls, a
    reasoning budget).
    """

    max_tokens: int | None = None
    temperature: float | None = None
    thinking: bool | Thinking | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Reply:
    """The model's whole reply, last in the stream: one assistant
    message, with the model that wrote it and its usage."""

    kind: Literal["Reply"] = "Reply"
    message: Message


ProviderEvent = TextDelta | ThinkingDelta | Reply
"""What a provider streams: text and reasoning deltas as they arrive,
then one :class:`Reply`."""


@runtime_checkable
class Provider(Protocol):
    """A model, as agex calls it."""

    @property
    def name(self) -> str:
        """The model, as a person would name it (``anthropic:claude-...``)."""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        settings: Settings = Settings(),
    ) -> AsyncIterator[ProviderEvent]:
        """Stream the reply to ``messages``: deltas, then one
        :class:`Reply`. A failed request raises."""
        ...

    def transient(self, error: BaseException) -> bool:
        """Whether a request that raised ``error`` is worth resuming:
        the provider was overloaded, rate-limited or briefly down, so
        the same request may succeed later. A run that hits one is
        interrupted, not failed."""
        ...
