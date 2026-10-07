"""A provider over ``pydantic_ai.direct``, and the mapping between
agex's messages and pydantic-ai's.

pydantic-ai is the transport only: a request is built from agex's
messages with :func:`to_messages`, streamed
through ``model_request_stream``, and the reply read back with
:func:`from_response`. Its ``Agent`` is not used.

The mapping keeps what a later request needs: tool-call ids (so a
result reaches its call), reasoning with its signature, provider and
details (so it can be sent back), and usage with its cache tokens.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from nontainer.turns import TextDelta, ThinkingDelta
from pydantic_ai import messages as pai
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..record import (
    Message,
    Part,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    Usage,
    new_id,
)
from . import ProviderEvent, Reply, Settings, ToolSpec

__all__ = ["PydanticAIProvider", "from_messages", "from_response", "to_messages"]

_logger = logging.getLogger(__name__)

#: HTTP statuses below 500 that a later attempt may get past.
TRANSIENT_STATUS = frozenset({408, 409, 425, 429})

#: What the provider SDKs name a provider they could not reach.
_UNREACHABLE_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})


def _is_httpx_transport(error: BaseException) -> bool:
    # by name: httpx comes with a provider's SDK, not with agex
    return any(
        cls.__name__ == "TransportError" and cls.__module__.startswith("httpx")
        for cls in type(error).__mro__
    )


def _unreachable(error: BaseException) -> bool:
    """Whether ``error`` (or what caused it) says the provider could not
    be reached: the SDKs' connection and timeout errors, httpx's
    transport errors, or the builtin connection and timeout errors."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        if type(current).__name__ in _UNREACHABLE_NAMES or _is_httpx_transport(current):
            return True
        current = current.__cause__ or current.__context__
    return False


# -- agex messages to pydantic-ai ---------------------------------------------------


def _request_parts(message: Message) -> list[pai.ModelRequestPart]:
    if message.role == "system":
        return [
            pai.SystemPromptPart(content=p.text)
            for p in message.parts
            if isinstance(p, Text)
        ]
    if message.role == "user":
        texts = [p.text for p in message.parts if isinstance(p, Text)]
        return [pai.UserPromptPart(content=texts[0] if len(texts) == 1 else texts)]
    return [
        pai.ToolReturnPart(
            tool_name=p.name,
            content=p.content,
            tool_call_id=p.call_id,
            outcome="failed" if p.is_error else "success",
        )
        for p in message.parts
        if isinstance(p, ToolResult)
    ]


def _response_part(part: Part) -> pai.ModelResponsePart | None:
    if isinstance(part, Text):
        return pai.TextPart(content=part.text)
    if isinstance(part, Thinking):
        return pai.ThinkingPart(
            content=part.text,
            signature=part.signature,
            provider_name=part.provider,
            provider_details=part.details,
        )
    if isinstance(part, ToolCall):
        return pai.ToolCallPart(
            tool_name=part.name, args=dict(part.args), tool_call_id=part.call_id
        )
    return None


def to_messages(messages: Sequence[Message]) -> list[pai.ModelMessage]:
    """agex messages as pydantic-ai's: each assistant message one
    ``ModelResponse``, and the system, user and tool messages between
    them gathered into one ``ModelRequest``."""
    out: list[pai.ModelMessage] = []
    pending: list[pai.ModelRequestPart] = []
    for message in messages:
        if message.role != "assistant":
            pending.extend(_request_parts(message))
            continue
        if pending:
            out.append(pai.ModelRequest(parts=pending))
            pending = []
        parts = [p for p in map(_response_part, message.parts) if p is not None]
        usage = message.usage or Usage()
        out.append(
            pai.ModelResponse(
                parts=parts,
                usage=RequestUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                ),
                model_name=message.model,
                provider_name=message.provider,
            )
        )
    if pending:
        out.append(pai.ModelRequest(parts=pending))
    return out


# -- pydantic-ai to agex messages ---------------------------------------------------


def from_response(response: pai.ModelResponse, *, id: str | None = None) -> Message:
    """A model reply as an assistant message. Parts agex does not keep
    (native tool calls, files) are left out, and logged."""
    parts: list[Part] = []
    for part in response.parts:
        if isinstance(part, pai.TextPart):
            parts.append(Text(text=part.content))
        elif isinstance(part, pai.ThinkingPart):
            parts.append(
                Thinking(
                    text=part.content,
                    signature=part.signature,
                    provider=part.provider_name,
                    details=dict(part.provider_details)
                    if part.provider_details
                    else None,
                )
            )
        elif isinstance(part, pai.ToolCallPart):
            parts.append(
                ToolCall(
                    call_id=part.tool_call_id,
                    name=part.tool_name,
                    args=part.args_as_dict(),
                )
            )
        else:
            _logger.warning("left out a %s part of a model reply", part.part_kind)
    usage = response.usage
    return Message(
        id=id or new_id(),
        role="assistant",
        parts=tuple(parts),
        model=response.model_name,
        provider=response.provider_name,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        ),
    )


def from_messages(messages: Sequence[pai.ModelMessage]) -> list[Message]:
    """pydantic-ai messages as agex's, each with a fresh id, in order: a
    system prompt and a user prompt each become one message, a run of
    consecutive tool results one tool message, and each response one
    assistant message."""
    out: list[Message] = []
    results: list[Part] = []

    def flush() -> None:
        if results:
            out.append(Message(id=new_id(), role="tool", parts=tuple(results)))
            results.clear()

    for message in messages:
        if isinstance(message, pai.ModelResponse):
            flush()
            out.append(from_response(message))
            continue
        for part in message.parts:
            if isinstance(part, pai.ToolReturnPart):
                results.append(
                    ToolResult(
                        call_id=part.tool_call_id,
                        name=part.tool_name,
                        content=part.content
                        if isinstance(part.content, str)
                        else part.model_response_str(),
                        is_error=part.outcome == "failed",
                    )
                )
                continue
            if isinstance(part, pai.RetryPromptPart) and part.tool_name:
                results.append(
                    ToolResult(
                        call_id=part.tool_call_id,
                        name=part.tool_name,
                        content=part.model_response(),
                        is_error=True,
                    )
                )
                continue
            # anything else follows the results before it
            flush()
            if isinstance(part, pai.SystemPromptPart):
                out.append(
                    Message(
                        id=new_id(), role="system", parts=(Text(text=part.content),)
                    )
                )
            elif isinstance(part, pai.UserPromptPart):
                content = part.content
                texts = (
                    [content]
                    if isinstance(content, str)
                    else [c for c in content if isinstance(c, str)]
                )
                out.append(
                    Message(
                        id=new_id(),
                        role="user",
                        parts=tuple(Text(text=t) for t in texts),
                    )
                )
            elif isinstance(part, pai.RetryPromptPart):
                out.append(
                    Message(
                        id=new_id(),
                        role="user",
                        parts=(Text(text=part.model_response()),),
                    )
                )
        flush()
    return out


# -- the provider -------------------------------------------------------------------


def _deltas(event: pai.ModelResponseStreamEvent) -> list[ProviderEvent]:
    """The text and reasoning a stream event adds. A part's first chunk
    arrives with its start event, the rest as deltas; tool-call
    arguments are not streamed out (the loop reports the call when it
    runs it)."""
    if isinstance(event, pai.PartStartEvent):
        part = event.part
        if isinstance(part, pai.TextPart) and part.content:
            return [TextDelta(text=part.content)]
        if isinstance(part, pai.ThinkingPart) and part.content:
            return [ThinkingDelta(text=part.content)]
    elif isinstance(event, pai.PartDeltaEvent):
        delta = event.delta
        if isinstance(delta, pai.TextPartDelta) and delta.content_delta:
            return [TextDelta(text=delta.content_delta)]
        if isinstance(delta, pai.ThinkingPartDelta) and delta.content_delta:
            return [ThinkingDelta(text=delta.content_delta)]
    return []


def _model_settings(settings: Settings) -> ModelSettings | None:
    merged: dict[str, Any] = dict(settings.extra)
    if settings.max_tokens is not None:
        merged["max_tokens"] = settings.max_tokens
    if settings.temperature is not None:
        merged["temperature"] = settings.temperature
    return ModelSettings(**merged) if merged else None


class PydanticAIProvider:
    """A :class:`~agex.providers.Provider` over ``pydantic_ai.direct``.

    ``model`` is a pydantic-ai ``Model`` or a model name pydantic-ai
    knows (``"anthropic:claude-sonnet-5-5"``, ``"openai:gpt-5"``), whose
    provider's extra must be installed (``agex[anthropic]``).
    """

    def __init__(self, model: Model | str) -> None:
        self.model: Model = infer_model(model) if isinstance(model, str) else model

    @property
    def name(self) -> str:
        return f"{self.model.system}:{self.model.model_name}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"

    def transient(self, error: BaseException) -> bool:
        """An error the provider may get past: an HTTP timeout (408), a
        conflict or too-early (409, 425), a rate limit (429) or any
        server-side error (5xx, 529 included), or a provider that could
        not be reached at all (a connection refused, dropped or timed
        out). Anything else, a refused request or a content filter, is
        not."""
        if isinstance(error, ModelHTTPError):
            code = error.status_code
            return code in TRANSIENT_STATUS or code >= 500
        return _unreachable(error)

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        settings: Settings = Settings(),
    ) -> AsyncIterator[ProviderEvent]:
        parameters = ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters,
                )
                for t in tools
            ],
            allow_text_output=True,
        )
        async with model_request_stream(
            self.model,
            to_messages(messages),
            model_settings=_model_settings(settings),
            model_request_parameters=parameters,
        ) as response:
            async for event in response:
                for delta in _deltas(event):
                    yield delta
            reply = response.get()
        yield Reply(message=from_response(reply))
