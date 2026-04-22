"""OpenAI-compatible LLM client using pyfetch for Pyodide/browser environments.

Works with OpenRouter, OpenAI, and any OpenAI-compatible API endpoint.
Uses pyodide.http.pyfetch for HTTP transport instead of the openai SDK,
enabling direct browser-to-API calls without a server proxy.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Iterator, List

from agex.agent.events import Event
from agex.llm.adapter import DefaultPyfetchAdapter, FetchAdapter
from agex.llm.core import LLM, TokenChunk
from agex.llm.formats import WireFormat, XmlWireFormat

CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}

# Flip to True to print raw SSE text deltas (pre-XML-tokenization) as they
# arrive from the API.  Useful for debugging what the model is actually
# producing vs what the XML tokenizer extracts.
DEBUG_RAW_STREAM = False


def _is_network_error(exc: Exception) -> bool:
    """Check if an exception is a transient network error worth retrying."""
    name = type(exc).__name__
    # Pyodide wraps JS TypeError("network error") as JsException
    if name == "JsException" and "network error" in str(exc).lower():
        return True
    # aiohttp network errors (non-Pyodide fallback)
    if name in ("ClientError", "ServerDisconnectedError", "ClientOSError"):
        return True
    return False


def _format_message_for_openai(message: dict[str, Any], *, cache: bool = False) -> dict:
    """Convert generic message dict to OpenAI's format (images as data URIs)."""
    if isinstance(message.get("content"), list):
        content_parts = []
        for part in message["content"]:
            if part["type"] == "text":
                content_parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{part['image_data']}"
                        },
                    }
                )
        if cache and content_parts:
            content_parts[-1]["cache_control"] = CACHE_CONTROL
        return {"role": message["role"], "content": content_parts}
    if cache:
        return {
            "role": message["role"],
            "content": [
                {
                    "type": "text",
                    "text": message["content"],
                    "cache_control": CACHE_CONTROL,
                }
            ],
        }
    return message


class PyfetchOpenAI(LLM):
    """OpenAI-compatible client using pyfetch (for Pyodide/browser).

    Supports OpenRouter, OpenAI, and any OpenAI-compatible endpoint.

    Args:
        model: Model identifier (e.g. "anthropic/claude-sonnet-4" for OpenRouter).
        api_key: API key for the endpoint.
        base_url: API base URL. Defaults to OpenRouter.
        timeout_seconds: Per-request timeout in seconds.
        **kwargs: Extra parameters forwarded to the completions request body
            (e.g. temperature, max_tokens).
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: str = "",
        base_url: str | None = None,
        timeout_seconds: float = 90.0,
        *,
        fetch_adapter: FetchAdapter | None = None,
        wire_format: WireFormat | None = None,
        **kwargs,
    ):
        kwargs.pop("provider", None)
        self._app_url = kwargs.pop("app_url", None)
        self._app_title = kwargs.pop("app_title", None)
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._kwargs = kwargs
        # Transport seam. When empty api_key is paired with a custom
        # adapter, the adapter is expected to inject auth headers on the
        # way out (e.g., a JS bridge that reads the key from localStorage).
        self._adapter: FetchAdapter = fetch_adapter or DefaultPyfetchAdapter()
        self._wire_format: WireFormat = wire_format or XmlWireFormat()

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        if self._app_url:
            h["HTTP-Referer"] = self._app_url
        if self._app_title:
            h["X-Title"] = self._app_title
        return h

    # -- LLM interface -------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "PyfetchOpenAI"

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    def dump_config(self) -> dict[str, Any]:
        return {
            "provider": "pyfetch_openai",
            "model": self._model,
            "base_url": self._base_url,
            "timeout_seconds": self._timeout_seconds,
            "app_url": self._app_url,
            "app_title": self._app_title,
            **self._kwargs,
        }

    # -- Streaming -----------------------------------------------------------

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """Not supported — pyfetch is async-only.  Use acomplete_stream."""
        raise NotImplementedError(
            "PyfetchOpenAI requires async. Use acomplete_stream() or acomplete()."
        )

    _STREAM_MAX_RETRIES = 2

    async def acomplete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """Stream tokens from an OpenAI-compatible endpoint via pyfetch.

        Retries on network errors (e.g. connection drops on mobile).
        """
        request_kwargs = {**self._kwargs, **kwargs}

        messages_dicts = self._wire_format.render_events(events)
        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        system_msg = _format_message_for_openai(
            {"role": "system", "content": system_with_format}, cache=True
        )
        # Place cache breakpoint on the second-to-last message (the end of
        # the previous turn's context).  The last message is always new, so
        # caching it would never yield a hit.  With the breakpoint one step
        # back, the entire prefix (system + history) gets cached across turns.
        cache_idx = len(messages_dicts) - 2
        conversation = [
            _format_message_for_openai(m, cache=(i == cache_idx))
            for i, m in enumerate(messages_dicts)
        ]
        full_messages = [system_msg] + conversation

        body = {
            "model": self._model,
            "messages": full_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **request_kwargs,
        }

        headers = self._headers()
        url = f"{self._base_url}/chat/completions"

        for attempt in range(self._STREAM_MAX_RETRIES):
            try:
                async for token in self._stream_once(url, body, headers):
                    yield token
                return  # success
            except Exception as exc:
                is_network = _is_network_error(exc)
                if not is_network or attempt + 1 >= self._STREAM_MAX_RETRIES:
                    raise
                # Brief pause before retry
                await asyncio.sleep(1)

    async def _stream_once(
        self,
        url: str,
        body: dict,
        headers: dict,
    ) -> AsyncIterator[TokenChunk]:
        """Single streaming attempt — separated for retry logic."""
        from agex.llm.sse import parse_sse_events

        response = self._adapter.fetch_stream(url, headers=headers, body=body)

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        sse_iter = parse_sse_events(response)

        def _update_usage(data: dict) -> None:
            usage = data.get("usage")
            if usage:
                usage_holder["input_tokens"] = usage.get("prompt_tokens")
                usage_holder["output_tokens"] = usage.get("completion_tokens")

        async def raw_chunks() -> AsyncIterator[str]:
            async for payload in sse_iter:
                if not payload.strip():
                    continue
                data = json.loads(payload)
                _update_usage(data)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if DEBUG_RAW_STREAM:
                            print(f"[openai raw] {content!r}")
                        yield content

        async for token in self._wire_format.aparse_text_stream(raw_chunks()):
            yield token

        # XML tokenizer may stop early (after </PYTHON> or </TERMINAL>).
        # Drain remaining SSE events briefly to capture the usage chunk
        # — but bound the wait tightly so a provider that stalls or
        # keeps streaming (e.g. some OpenRouter routes not honoring
        # stream_options, or a model that keeps generating past our
        # stop point) can't hang the whole chat turn.
        async def _drain() -> None:
            async for payload in sse_iter:
                if not payload.strip():
                    continue
                data = json.loads(payload)
                _update_usage(data)
                if usage_holder["input_tokens"] is not None:
                    return

        if usage_holder["input_tokens"] is None:
            try:
                await asyncio.wait_for(_drain(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                # Best-effort — partial or missing usage is acceptable.
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
        is_multimodal, processed = self._prepare_summarization_content(content)

        if is_multimodal:
            full_messages = [{"role": "system", "content": system}] + processed
        else:
            full_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": processed},
            ]

        body = {
            "model": self._model,
            "messages": [_format_message_for_openai(m) for m in full_messages],
            **request_kwargs,
        }

        headers = self._headers()

        response_data = await self._adapter.fetch_json(
            f"{self._base_url}/chat/completions",
            headers=headers,
            body=body,
        )

        return response_data["choices"][0]["message"]["content"] or ""
