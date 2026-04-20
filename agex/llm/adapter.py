"""Transport-layer adapter for LLM HTTP calls.

The ``Pyfetch{OpenAI,Anthropic}`` clients delegate their network transport
to a :class:`FetchAdapter`, keeping request construction and response
parsing (SSE, retries, usage tracking) separate from the actual HTTP
machinery.

This lets callers in specific environments inject a custom transport —
for example, a JS-bridge adapter that routes calls through a main-thread
``fetch`` so the API key can live in JS rather than Python scope.
"""

import codecs
import json
import sys
from abc import ABC, abstractmethod
from typing import AsyncIterator


class FetchAdapter(ABC):
    """Transport seam for LLM HTTP calls.

    Subclass and override to route LLM traffic through a custom transport.
    The default implementation (:class:`DefaultPyfetchAdapter`) uses
    ``pyodide.http.pyfetch`` in Pyodide and ``aiohttp`` elsewhere.
    """

    @abstractmethod
    async def fetch_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict,
    ) -> dict:
        """POST ``body`` as JSON and return the parsed JSON response.

        Must raise ``RuntimeError`` with a descriptive message on HTTP >= 400.
        """
        ...

    @abstractmethod
    async def fetch_stream(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict,
    ) -> AsyncIterator[str]:
        """POST ``body`` as JSON and return an async iterator of text chunks.

        Chunks are UTF-8-decoded text. The caller handles SSE parsing
        downstream. Must raise ``RuntimeError`` on HTTP >= 400 before
        yielding.
        """
        ...


class DefaultPyfetchAdapter(FetchAdapter):
    """Default transport: ``pyfetch`` in Pyodide, ``aiohttp`` elsewhere.

    This is a direct extraction of the transport code previously duplicated
    in ``PyfetchOpenAI`` and ``PyfetchAnthropic`` — behavior is unchanged
    from what those clients did inline.
    """

    async def fetch_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict,
    ) -> dict:
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
                        if resp.status >= 400:
                            try:
                                error_body = await resp.json()
                                error_msg = error_body.get("error", {}).get(
                                    "message", str(error_body)
                                )
                            except Exception:
                                error_msg = f"HTTP {resp.status}"
                            raise RuntimeError(
                                f"API error ({resp.status}): {error_msg}"
                            )
                        return await resp.json()
            except ImportError:
                raise RuntimeError(
                    "DefaultPyfetchAdapter requires pyodide (emscripten) or "
                    "aiohttp installed."
                )

    async def fetch_stream(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict,
    ) -> AsyncIterator[str]:
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
                text = text_decoder.decode(result.value, {"stream": True})
                yield text
            # Flush pending bytes from any incomplete multi-byte sequence
            # retained across the last call with stream=True. If the stream
            # happens to end mid-multibyte (rare, e.g. truncated network),
            # this surfaces the remaining bytes rather than silently dropping.
            final_text = text_decoder.decode()
            if final_text:
                yield final_text
        else:
            try:
                import aiohttp

                decoder = codecs.getincrementaldecoder("utf-8")()
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=body, headers=headers) as resp:
                        if resp.status >= 400:
                            try:
                                error_body = await resp.json()
                                error_msg = error_body.get("error", {}).get(
                                    "message", str(error_body)
                                )
                            except Exception:
                                error_msg = f"HTTP {resp.status}"
                            raise RuntimeError(
                                f"API error ({resp.status}): {error_msg}"
                            )
                        async for chunk in resp.content.iter_any():
                            text = decoder.decode(chunk, False)
                            if text:
                                yield text
                        final_text = decoder.decode(b"", True)
                        if final_text:
                            yield final_text
            except ImportError:
                raise RuntimeError(
                    "DefaultPyfetchAdapter requires pyodide (emscripten) or "
                    "aiohttp installed."
                )
