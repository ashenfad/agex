"""OpenAI-compatible LLM client using pyfetch for Pyodide/browser environments.

Works with OpenRouter, OpenAI, and any OpenAI-compatible API endpoint.
Uses pyodide.http.pyfetch for HTTP transport instead of the openai SDK,
enabling direct browser-to-API calls without a server proxy.
"""

import codecs
import json
import sys
from typing import Any, AsyncIterator, Iterator, List

from agex.agent.events import Event
from agex.llm.core import LLM, TokenChunk
from agex.llm.xml import XML_FORMAT_PRIMER

CACHE_CONTROL = {"type": "ephemeral"}


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
        **kwargs,
    ):
        kwargs.pop("provider", None)
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._kwargs = kwargs

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

    async def acomplete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """Stream tokens from an OpenAI-compatible endpoint via pyfetch."""
        from agex.llm.sse import parse_sse_events
        from agex.llm.xml import atokenize_xml_stream
        from agex.render.xml import render_events_as_xml

        request_kwargs = {**self._kwargs, **kwargs}

        messages_dicts = render_events_as_xml(events)
        system_with_format = f"{system}\n\n{XML_FORMAT_PRIMER}"
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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        response = self._pyfetch_stream(
            f"{self._base_url}/chat/completions",
            body=body,
            headers=headers,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        # Wrap SSE stream so we can drain remaining events after XML stops
        sse_iter = parse_sse_events(response)

        def _update_usage(data: dict) -> None:
            usage = data.get("usage")
            if usage:
                usage_holder["input_tokens"] = usage.get("prompt_tokens")
                usage_holder["output_tokens"] = usage.get("completion_tokens")

        async def raw_chunks() -> AsyncIterator[str]:
            async for payload in sse_iter:
                data = json.loads(payload)
                _update_usage(data)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

        async for token in atokenize_xml_stream(raw_chunks()):
            yield token

        # XML tokenizer may stop early (after </PYTHON> or </TERMINAL>).
        # Drain remaining SSE events to capture the usage chunk.
        if usage_holder["input_tokens"] is None:
            async for payload in sse_iter:
                data = json.loads(payload)
                _update_usage(data)
                if usage_holder["input_tokens"] is not None:
                    break

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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        response_data = await self._pyfetch_json(
            f"{self._base_url}/chat/completions",
            body=body,
            headers=headers,
        )

        return response_data["choices"][0]["message"]["content"] or ""

    # -- Transport -----------------------------------------------------------

    @staticmethod
    async def _pyfetch_stream(
        url: str,
        body: dict,
        headers: dict,
    ) -> AsyncIterator[str]:
        """POST and return an async iterator of text chunks from the SSE stream."""
        if sys.platform == "emscripten":
            from pyodide.http import pyfetch

            resp = await pyfetch(
                url,
                method="POST",
                headers=headers,
                body=json.dumps(body),
            )
            if resp.status >= 400:
                try:
                    error_body = await resp.json()
                    error_msg = error_body.get("error", {}).get(
                        "message", str(error_body)
                    )
                except Exception:
                    error_msg = f"HTTP {resp.status}"
                raise RuntimeError(f"API error ({resp.status}): {error_msg}")
            js_response = resp.js_response
            reader = js_response.body.getReader()

            from js import TextDecoder

            text_decoder = TextDecoder.new("utf-8")

            while True:
                result = await reader.read()
                if result.done:
                    break
                # result.value is a Uint8Array
                text = text_decoder.decode(result.value, {"stream": True})
                yield text
        else:
            # Non-Pyodide fallback (for testing with aiohttp)
            try:
                import aiohttp

                decoder = codecs.getincrementaldecoder("utf-8")()
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=body, headers=headers) as resp:
                        resp.raise_for_status()
                        async for chunk in resp.content.iter_any():
                            text = decoder.decode(chunk, False)
                            if text:
                                yield text
            except ImportError:
                raise RuntimeError(
                    "PyfetchOpenAI requires pyodide (emscripten) or aiohttp installed."
                )

    @staticmethod
    async def _pyfetch_json(
        url: str,
        body: dict,
        headers: dict,
    ) -> dict:
        """POST and return parsed JSON response."""
        if sys.platform == "emscripten":
            from pyodide.http import pyfetch

            resp = await pyfetch(
                url,
                method="POST",
                headers=headers,
                body=json.dumps(body),
            )
            if resp.status >= 400:
                try:
                    error_body = await resp.json()
                    error_msg = error_body.get("error", {}).get(
                        "message", str(error_body)
                    )
                except Exception:
                    error_msg = f"HTTP {resp.status}"
                raise RuntimeError(f"API error ({resp.status}): {error_msg}")
            return await resp.json()
        else:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=body, headers=headers) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except ImportError:
                raise RuntimeError(
                    "PyfetchOpenAI requires pyodide (emscripten) or aiohttp installed."
                )
