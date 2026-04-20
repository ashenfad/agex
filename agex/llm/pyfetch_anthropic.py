"""Anthropic LLM client using pyfetch for Pyodide/browser environments.

Calls the Anthropic Messages API directly from the browser via
``pyodide.http.pyfetch``, enabling direct browser-to-API calls without a
server proxy.  Requires the ``anthropic-dangerous-direct-browser-access``
header (supported by Anthropic for trusted browser contexts).
"""

import asyncio
import json
from typing import Any, AsyncIterator, Iterator, List

from agex.agent.events import Event
from agex.llm.adapter import DefaultPyfetchAdapter, FetchAdapter
from agex.llm.core import LLM, TokenChunk
from agex.llm.xml import XML_FORMAT_PRIMER

ANTHROPIC_VERSION = "2023-06-01"
CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}
MAX_TOKENS = 2**14

# Flip to True to print raw SSE text deltas (pre-XML-tokenization) as they
# arrive from the Anthropic API.  Useful for debugging what the model is
# actually producing vs what the XML tokenizer extracts.
DEBUG_RAW_STREAM = False


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

        Retries on network errors (e.g. connection drops on mobile).
        """
        from agex.render.xml import render_events_as_xml

        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" not in request_kwargs:
            request_kwargs["max_tokens"] = MAX_TOKENS

        messages_dicts = render_events_as_xml(events)

        # System message with XML format primer, cached.
        system_with_format = f"{system}\n\n{XML_FORMAT_PRIMER}"
        system_blocks = [
            {
                "type": "text",
                "text": system_with_format,
                "cache_control": CACHE_CONTROL,
            }
        ]

        # Place cache breakpoint on second-to-last message (end of previous
        # turn's context). The last message is always new, so caching it
        # wouldn't yield a hit.
        cache_idx = len(messages_dicts) - 2
        conversation = [
            _format_message_for_anthropic(m, cache=(i == cache_idx))
            for i, m in enumerate(messages_dicts)
        ]
        # No assistant prefill — letting the model produce the whole response
        # from scratch gives it better adherence to the format primer (closing
        # tags, no skipping <THINKING>, etc.).  If the model starts adding
        # preamble like "Sure, here's my response:", reintroduce prefill or
        # add a stop_sequences=["</PYTHON>", "</TERMINAL>"] instead.

        body = {
            "model": self._model,
            "system": system_blocks,
            "messages": conversation,
            "stream": True,
            **request_kwargs,
        }

        headers = self._headers()
        url = f"{self._base_url}/messages"

        for attempt in range(self._STREAM_MAX_RETRIES):
            try:
                async for token in self._stream_once(url, body, headers):
                    yield token
                return  # success
            except Exception as exc:
                is_network = _is_network_error(exc)
                if not is_network or attempt + 1 >= self._STREAM_MAX_RETRIES:
                    raise
                import asyncio

                await asyncio.sleep(1)

    async def _stream_once(
        self,
        url: str,
        body: dict,
        headers: dict,
    ) -> AsyncIterator[TokenChunk]:
        """Single streaming attempt — separated for retry logic."""
        from agex.llm.sse import parse_sse_events
        from agex.llm.xml import atokenize_xml_stream

        response = self._adapter.fetch_stream(url, headers=headers, body=body)

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        sse_iter = parse_sse_events(response)

        def _total_input(u: dict) -> int | None:
            # With prompt caching, Anthropic splits input tokens into three
            # buckets — sum them for the true total.
            if "input_tokens" not in u:
                return None
            return (
                int(u.get("input_tokens") or 0)
                + int(u.get("cache_creation_input_tokens") or 0)
                + int(u.get("cache_read_input_tokens") or 0)
            )

        def _update_usage(data: dict) -> None:
            # message_start: usage is nested under message
            msg = data.get("message")
            if msg and isinstance(msg.get("usage"), dict):
                u = msg["usage"]
                total_in = _total_input(u)
                if total_in is not None:
                    usage_holder["input_tokens"] = total_in
                if "output_tokens" in u:
                    usage_holder["output_tokens"] = u["output_tokens"]
            # message_delta: usage at top level, carries final output_tokens
            if isinstance(data.get("usage"), dict):
                u = data["usage"]
                total_in = _total_input(u)
                if total_in is not None:
                    usage_holder["input_tokens"] = total_in
                if "output_tokens" in u:
                    usage_holder["output_tokens"] = u["output_tokens"]

        async def raw_chunks() -> AsyncIterator[str]:
            # Note: Anthropic's raw content_block_delta events include the
            # prefill text, so we don't need to yield it manually (doing so
            # would double the prefill and break XML tokenization).
            async for payload in sse_iter:
                if not payload.strip():
                    continue
                data = json.loads(payload)
                evt_type = data.get("type")
                if evt_type == "message_start":
                    _update_usage(data)
                elif evt_type == "content_block_start":
                    if DEBUG_RAW_STREAM:
                        idx = data.get("index")
                        block = data.get("content_block", {})
                        print(
                            f"[anthropic block_start] index={idx} "
                            f"type={block.get('type')}"
                        )
                elif evt_type == "content_block_stop":
                    if DEBUG_RAW_STREAM:
                        print(f"[anthropic block_stop] index={data.get('index')}")
                elif evt_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text")
                        if text:
                            if DEBUG_RAW_STREAM:
                                idx = data.get("index")
                                print(f"[anthropic raw idx={idx}] {text!r}")
                            yield text
                elif evt_type == "message_delta":
                    _update_usage(data)
                elif evt_type == "error":
                    err = data.get("error", {})
                    raise RuntimeError(
                        f"Anthropic stream error: {err.get('message', err)}"
                    )

        async for token in atokenize_xml_stream(raw_chunks()):
            yield token

        # XML tokenizer may stop early (after </PYTHON> or </TERMINAL>).
        # Drain remaining SSE events to capture final usage — but bound
        # the wait tightly so we don't hang on a model that keeps
        # generating past our stop point.  Anthropic models sometimes
        # hallucinate trailing content (e.g. a fake next user turn),
        # and without a timeout the drain waits for every single
        # byte up to max_tokens before yielding the final done token.
        async def _drain() -> None:
            async for payload in sse_iter:
                if not payload.strip():
                    continue
                data = json.loads(payload)
                if data.get("type") == "message_stop":
                    return
                if data.get("type") in ("message_start", "message_delta"):
                    _update_usage(data)

        try:
            await asyncio.wait_for(_drain(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            # Best-effort — partial usage is acceptable.  The orphaned
            # SSE reader will be garbage-collected or closed when the
            # underlying HTTP response finishes.
            pass

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
