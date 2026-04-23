"""Anthropic LLM client using pyfetch for Pyodide/browser environments.

Calls the Anthropic Messages API directly from the browser via
``pyodide.http.pyfetch``, enabling direct browser-to-API calls without a
server proxy.  Requires the ``anthropic-dangerous-direct-browser-access``
header (supported by Anthropic for trusted browser contexts).
"""

import json
from typing import Any, AsyncIterator, Iterator, List

from agex.agent.events import Event
from agex.llm.adapter import DefaultPyfetchAdapter, FetchAdapter
from agex.llm.core import LLM, TokenChunk
from agex.llm.formats import ToolUseWireFormat, WireFormat
from agex.llm.formats.tool_use.anthropic_adapter import (
    apply_cache_control,
    atranslate_anthropic_stream_to_events,
    schemas_to_anthropic_tools,
    translate_messages_to_anthropic,
)

ANTHROPIC_VERSION = "2023-06-01"
CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}
MAX_TOKENS = 2**14


def _is_network_error(exc: Exception) -> bool:
    """Check if an exception is a transient network error worth retrying."""
    name = type(exc).__name__
    if name == "JsException" and "network error" in str(exc).lower():
        return True
    if name in ("ClientError", "ServerDisconnectedError", "ClientOSError"):
        return True
    return False


def _format_message_for_anthropic(
    message: dict[str, Any], *, cache: bool = False
) -> dict:
    """Convert generic message dict to Anthropic's format."""
    content_parts: list[dict] = []
    if isinstance(message.get("content"), list):
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

    if cache and content_parts:
        content_parts[-1]["cache_control"] = CACHE_CONTROL
    return {"role": message["role"], "content": content_parts}


class PyfetchAnthropic(LLM):
    """Anthropic client using pyfetch (for Pyodide/browser).

    Calls ``https://api.anthropic.com/v1/messages`` directly.  Requires the
    browser-origin flag (``anthropic-dangerous-direct-browser-access``).

    Args:
        model: Model identifier (e.g. "claude-sonnet-4-5").
        api_key: Anthropic API key.
        base_url: API base URL.  Defaults to the Anthropic API.
        timeout_seconds: Per-request timeout in seconds.
        **kwargs: Extra parameters forwarded to the messages request body
            (e.g. temperature, max_tokens).
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str = "",
        base_url: str | None = None,
        timeout_seconds: float = 90.0,
        *,
        fetch_adapter: FetchAdapter | None = None,
        wire_format: WireFormat | None = None,
        **kwargs,
    ):
        kwargs.pop("provider", None)
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._kwargs = kwargs
        # Transport seam. When empty api_key is paired with a custom
        # adapter, the adapter is expected to inject auth headers on the
        # way out (e.g., a JS bridge that reads the key from localStorage).
        self._adapter: FetchAdapter = fetch_adapter or DefaultPyfetchAdapter()
        self._wire_format: WireFormat = wire_format or ToolUseWireFormat()

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-dangerous-direct-browser-access": "true",
        }
        if self._api_key:
            h["x-api-key"] = self._api_key
        return h

    # -- LLM interface -------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "PyfetchAnthropic"

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    def dump_config(self) -> dict[str, Any]:
        return {
            "provider": "pyfetch_anthropic",
            "model": self._model,
            "base_url": self._base_url,
            "timeout_seconds": self._timeout_seconds,
            **self._kwargs,
        }

    # -- Streaming -----------------------------------------------------------

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """Not supported — pyfetch is async-only.  Use acomplete_stream."""
        raise NotImplementedError(
            "PyfetchAnthropic requires async. Use acomplete_stream() or acomplete()."
        )

    _STREAM_MAX_RETRIES = 2

    async def acomplete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """Stream tokens from the Anthropic Messages API via pyfetch.

        Dispatches on ``wire_format.tool_schema()``:

        - ``None`` → text-stream path (XML-in-text formats).
        - non-None → provider-native tool-calling path.

        Retries on network errors (e.g. connection drops on mobile).
        """
        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" not in request_kwargs:
            request_kwargs["max_tokens"] = MAX_TOKENS

        messages_dicts = self._wire_format.render_events(events)

        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        system_blocks = [
            {
                "type": "text",
                "text": system_with_format,
                "cache_control": CACHE_CONTROL,
            }
        ]

        tool_schemas = self._wire_format.tool_schema()
        translated = translate_messages_to_anthropic(messages_dicts)
        cache_idx = len(translated) - 2
        conversation = apply_cache_control(translated, cache_index=cache_idx, ttl="1h")
        body: dict[str, Any] = {
            "model": self._model,
            "system": system_blocks,
            "messages": conversation,
            "tools": schemas_to_anthropic_tools(tool_schemas),
            "stream": True,
            **request_kwargs,
        }

        headers = self._headers()
        url = f"{self._base_url}/messages"

        for attempt in range(self._STREAM_MAX_RETRIES):
            try:
                async for token in self._stream_once_tools(url, body, headers):
                    yield token
                return  # success
            except Exception as exc:
                is_network = _is_network_error(exc)
                if not is_network or attempt + 1 >= self._STREAM_MAX_RETRIES:
                    raise
                import asyncio

                await asyncio.sleep(1)

    async def _stream_once_tools(
        self,
        url: str,
        body: dict,
        headers: dict,
    ) -> AsyncIterator[TokenChunk]:
        """Single tool-use streaming attempt.

        Reads Anthropic SSE ``data:`` payloads, routes each to the
        adapter which maps ``content_block_*`` / ``message_*`` events
        into :class:`ToolCallEvent`\\ s.
        """
        from agex.llm.sse import parse_sse_events

        response = self._adapter.fetch_stream(url, headers=headers, body=body)

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        sse_iter = parse_sse_events(response)

        async def event_dicts():
            async for payload in sse_iter:
                if not payload.strip():
                    continue
                data = json.loads(payload)
                if data.get("type") == "error":
                    err = data.get("error", {})
                    raise RuntimeError(
                        f"Anthropic stream error: {err.get('message', err)}"
                    )
                yield data

        tool_events = atranslate_anthropic_stream_to_events(
            event_dicts(), usage_holder=usage_holder
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

    # -- Summarize -----------------------------------------------------------

    async def summarize(self, system: str, content: str | List[Event], **kwargs) -> str:
        """Text generation via pyfetch (async-only)."""
        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" not in request_kwargs:
            request_kwargs["max_tokens"] = MAX_TOKENS

        is_multimodal, processed = self._prepare_summarization_content(content)

        if is_multimodal:
            conversation = [_format_message_for_anthropic(m) for m in processed]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": processed}],
                }
            ]

        body = {
            "model": self._model,
            "system": system,
            "messages": conversation,
            **request_kwargs,
        }

        headers = self._headers()

        response_data = await self._adapter.fetch_json(
            f"{self._base_url}/messages",
            headers=headers,
            body=body,
        )

        # Concatenate text parts from content blocks.
        texts: list[str] = []
        for block in response_data.get("content") or []:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "".join(texts)
