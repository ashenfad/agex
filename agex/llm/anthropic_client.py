from typing import Any, AsyncIterator, Iterator, List

import anthropic
from anthropic.types import TextBlockParam

from agex.agent.events import Event
from agex.llm.core import (
    LLM,
    TokenChunk,
)
from agex.llm.formats import ToolUseWireFormat, WireFormat
from agex.llm.formats.tool_use.anthropic_adapter import (
    apply_cache_control,
    atranslate_anthropic_stream_to_events,
    schemas_to_anthropic_tools,
    translate_anthropic_stream_to_events,
    translate_messages_to_anthropic,
)

# Define keys for client setup vs. completion
CLIENT_CONFIG_KEYS = {"api_key", "timeout", "max_retries"}
MAX_TOKENS = 2**14
CACHE_TTL = "1h"

# Default budget for Claude extended thinking when ``native_thinking``
# is on.  The API requires at least 1024 tokens; a couple of thousand
# gives the model real reasoning space without eating into completion
# budget (``MAX_TOKENS`` covers both).  Callers on constrained budgets
# can override by passing their own ``thinking`` kwarg.
_DEFAULT_THINKING_BUDGET = 2048


def _ensure_extended_thinking(
    request_kwargs: dict, budget_tokens: int = _DEFAULT_THINKING_BUDGET
) -> dict:
    """Enable Claude's extended thinking so the adapter captures
    ``thinking`` blocks natively instead of relying on narration-in-
    schema.  Callers can override by passing ``thinking=`` explicitly
    (including ``thinking=None`` to opt out entirely on non-reasoning
    Claude variants).
    """
    if "thinking" in request_kwargs:
        return request_kwargs
    return {
        **request_kwargs,
        "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
    }


def _ensure_tool_choice_any(request_kwargs: dict) -> dict:
    """Default ``tool_choice`` to ``{"type": "any"}`` so Claude is
    forced to call one of our tools each turn.  Tools are agex's real
    API; prose alone doesn't advance the task.

    Skipped when the caller opts into extended thinking (``thinking``
    kwarg) — Anthropic rejects ``tool_choice != auto`` in that mode —
    or when the caller already passed their own ``tool_choice``.
    """
    if "tool_choice" in request_kwargs:
        return request_kwargs
    if "thinking" in request_kwargs:
        return request_kwargs
    return {**request_kwargs, "tool_choice": {"type": "any"}}


def _with_cache(messages: list[dict]) -> list[dict]:
    messages[-1]["cache_control"] = {"type": "ephemeral", "ttl": CACHE_TTL}
    return messages


def _format_message_for_anthropic(
    is_last_message: bool, message: dict[str, Any]
) -> dict:
    """
    Convert generic message dict to Anthropic's format.

    Handles multimodal content (images) conversion.

    Note: All images are converted to PNG format by the rendering layer
    (serialize_image_to_base64) before reaching this function.
    """
    content_parts: list[dict] = []
    if isinstance(message.get("content"), list):
        # Multimodal message
        for part in message["content"]:
            if part["type"] == "text":
                content_parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                content_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": part["image_data"],
                        },
                    }
                )
    else:
        content_parts.append({"type": "text", "text": message["content"]})

    if is_last_message:
        _with_cache(content_parts)
    return {"role": message["role"], "content": content_parts}


class Anthropic(LLM):
    """Anthropic LLM provider using tool calling for structured outputs."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        timeout_seconds: float = 90.0,
        wire_format: WireFormat | None = None,
        **kwargs,
    ):
        kwargs.pop("provider", None)
        client_kwargs: dict[str, Any] = {}
        completion_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in CLIENT_CONFIG_KEYS:
                client_kwargs[key] = value
            else:
                completion_kwargs[key] = value

        # Wire timeout and disable SDK retries (agex handles retries)
        client_kwargs.setdefault("timeout", timeout_seconds)
        client_kwargs.setdefault("max_retries", 0)

        self._model = model
        self._kwargs = completion_kwargs
        self._timeout_seconds = timeout_seconds
        # Claude 4+ supports extended thinking natively; prefer that
        # over narration-in-schema so ``thinking`` isn't a parameter
        # the model can fill while leaving ``code`` empty.  Users on
        # older Claude (3.x) can opt out via ``ToolUseWireFormat()``.
        self._wire_format: WireFormat = wire_format or ToolUseWireFormat(
            native_thinking=True
        )
        self.client = anthropic.Anthropic(**client_kwargs)
        self.async_client = anthropic.AsyncAnthropic(**client_kwargs)

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds for each API call."""
        return self._timeout_seconds

    def dump_config(self) -> dict[str, Any]:
        return {
            "provider": "anthropic",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            **self._kwargs,
        }

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """Stream tokens from Anthropic via the provider-native tool-use
        path."""
        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" not in request_kwargs:
            request_kwargs["max_tokens"] = MAX_TOKENS

        messages_dicts = self._wire_format.render_events(events)
        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        system_block = TextBlockParam(
            type="text",
            text=system_with_format,
            cache_control={"type": "ephemeral", "ttl": CACHE_TTL},
        )
        yield from self._stream_tools(
            messages_dicts,
            system_block,
            request_kwargs,
            self._wire_format.tool_schema(),
        )

    def _stream_tools(
        self,
        messages_dicts: list[dict],
        system_block: TextBlockParam,
        request_kwargs: dict,
        tool_schemas: list[dict],
    ) -> Iterator[TokenChunk]:
        translated = translate_messages_to_anthropic(messages_dicts)
        # Cache breakpoint on second-to-last message (end of prior turn's
        # context).  The last message is always new — caching it never hits.
        cache_idx = len(translated) - 2
        conversation_messages = apply_cache_control(
            translated, cache_index=cache_idx, ttl=CACHE_TTL
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        # Enable extended thinking when the wire format is native —
        # this is what lets the adapter capture real thinking blocks
        # instead of relying on ``thinking`` as a schema parameter.
        # Must run before ``_ensure_tool_choice_any`` so the helper
        # sees ``thinking`` already in kwargs and skips the (API-
        # incompatible) ``tool_choice=any`` force.
        if getattr(self._wire_format, "native_thinking", False):
            request_kwargs = _ensure_extended_thinking(request_kwargs)

        # Tools are agex's real API — force a tool call each turn so
        # the loop always makes progress.  Extended thinking is
        # incompatible with ``tool_choice != auto``, so skip the force
        # when callers opt into ``thinking``.  User-supplied
        # ``tool_choice`` always wins.
        request_kwargs = _ensure_tool_choice_any(request_kwargs)

        with self.client.messages.stream(
            model=self._model,
            system=[system_block],
            messages=conversation_messages,  # type: ignore[arg-type]
            tools=schemas_to_anthropic_tools(tool_schemas),  # type: ignore[arg-type]
            **request_kwargs,
        ) as stream:
            tool_events = translate_anthropic_stream_to_events(
                iter(stream), usage_holder=usage_holder
            )
            yield from self._wire_format.parse_tool_stream(tool_events)

        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

    async def acomplete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """Async version of :meth:`complete_stream`."""
        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" not in request_kwargs:
            request_kwargs["max_tokens"] = MAX_TOKENS

        messages_dicts = self._wire_format.render_events(events)
        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        system_block = TextBlockParam(
            type="text",
            text=system_with_format,
            cache_control={"type": "ephemeral", "ttl": CACHE_TTL},
        )
        async for t in self._astream_tools(
            messages_dicts,
            system_block,
            request_kwargs,
            self._wire_format.tool_schema(),
        ):
            yield t

    async def _astream_tools(
        self,
        messages_dicts: list[dict],
        system_block: TextBlockParam,
        request_kwargs: dict,
        tool_schemas: list[dict],
    ) -> AsyncIterator[TokenChunk]:
        translated = translate_messages_to_anthropic(messages_dicts)
        cache_idx = len(translated) - 2
        conversation_messages = apply_cache_control(
            translated, cache_index=cache_idx, ttl=CACHE_TTL
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        if getattr(self._wire_format, "native_thinking", False):
            request_kwargs = _ensure_extended_thinking(request_kwargs)
        request_kwargs = _ensure_tool_choice_any(request_kwargs)

        async with self.async_client.messages.stream(
            model=self._model,
            system=[system_block],
            messages=conversation_messages,  # type: ignore[arg-type]
            tools=schemas_to_anthropic_tools(tool_schemas),  # type: ignore[arg-type]
            **request_kwargs,
        ) as stream:
            tool_events = atranslate_anthropic_stream_to_events(
                stream.__aiter__(), usage_holder=usage_holder
            )
            async for token in self._wire_format.aparse_tool_stream(tool_events):
                yield token

        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

    def summarize(self, system: str, content: str | List[Event], **kwargs) -> str:
        """Send a summarization request to Anthropic (text or events with multimodal)."""
        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Prepare content (text or events)
        is_multimodal, processed = self._prepare_summarization_content(content)

        if is_multimodal:
            # processed is messages list from events
            # Convert to Anthropic format
            conversation_messages = [
                _format_message_for_anthropic(index == len(processed) - 1, msg)
                for index, msg in enumerate(processed)
            ]
        else:
            # processed is plain text
            conversation_messages = [
                {"role": "user", "content": [{"type": "text", "text": processed}]}
            ]

        if "max_tokens" not in request_kwargs:
            request_kwargs["max_tokens"] = MAX_TOKENS

        response = self.client.messages.create(
            model=self._model,
            system=system,
            messages=conversation_messages,
            **request_kwargs,
        )
        # Concatenate text parts from content blocks
        texts: list[str] = []
        for block in response.content or []:
            if getattr(block, "type", None) == "text":
                texts.append(getattr(block, "text", ""))
        return "".join(texts)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Anthropic"
