"""OpenAI-compatible LLM client using pyfetch for Pyodide/browser environments.

Works with OpenRouter, OpenAI, and any OpenAI-compatible API endpoint.
Uses pyodide.http.pyfetch for HTTP transport instead of the openai SDK,
enabling direct browser-to-API calls without a server proxy.
"""

import asyncio
import hashlib
import json
from typing import Any, AsyncIterator, Iterator, List

from agex.agent.events import Event
from agex.llm.adapter import DefaultPyfetchAdapter, FetchAdapter
from agex.llm.core import LLM, TokenChunk
from agex.llm.formats import ToolUseWireFormat, WireFormat
from agex.llm.formats.tool_use.openai_adapter import (
    atranslate_openai_stream_to_events,
    schemas_to_openai_tools,
    translate_messages_to_openai,
)
from agex.llm.formats.tool_use.openai_responses_adapter import (
    atranslate_openai_responses_stream_to_events,
    schemas_to_openai_responses_tools,
    translate_messages_to_openai_responses,
)


def _is_reasoning_model(model: str) -> bool:
    """Mirrors the dispatch rule used by the SDK client — GPT-5 family
    and o1/o3 reasoning models use the Responses endpoint; everything
    else stays on Chat Completions.  Accepts OpenRouter-style
    ``openai/gpt-5-mini`` prefixes alongside plain ``gpt-5-mini``.
    """
    m = model.lower()
    if "/" in m:
        m = m.split("/", 1)[1]
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")


def _hash12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# Char-position checkpoints at which we hash the request prefix.
# Consecutive request lines can be diffed at each checkpoint to see
# where drift starts — e.g. if ck_4k is identical across two requests
# but ck_16k differs, the content between 4k and 16k chars is drifting.
_CACHE_CHECKPOINTS = (1024, 4096, 16384, 65536)


def _log_cache_diagnostics(
    body: dict, cache_idx_in_conv: int, usage_holder: dict
) -> None:
    """Emit one line per request capturing the prefix shape and the
    provider-reported cache hit.

    Hashes use ``sort_keys=False`` to match what the wire actually
    serializes (``json.dumps(body)`` preserves insertion order), so
    the hash reflects exactly what Anthropic/OpenRouter sees.  Dict
    iteration order in Python 3.7+ is insertion order, so stable
    construction gives stable hashes.

    Hashes are computed at multiple fixed character positions so two
    consecutive request lines can be diffed to pinpoint where any
    prefix drift starts — a stable ``ck_4k`` but diverging ``ck_16k``
    means the content between 4k and 16k chars is drifting.  A
    diverging ``sys_hash`` means the system message itself is drifting.

    Uses ``print`` rather than the ``logging`` module because Python's
    ``logger.info`` is a silent no-op without a handler, and the
    pyfetch clients are deployed into pages that don't configure one.
    Pyodide routes stdout to ``console.log``.
    """
    msgs = body.get("messages", [])
    sys_str = json.dumps(msgs[0]) if msgs else ""
    sys_hash = _hash12(sys_str) if sys_str else "<none>"

    prefix_idx = 1 + cache_idx_in_conv  # +1 to account for the system msg
    prefix_msgs = msgs[: prefix_idx + 1]
    prefix_str = json.dumps(prefix_msgs)
    prefix_hash = _hash12(prefix_str)

    # Hash the prefix at each checkpoint (no-op for checkpoints beyond
    # the prefix length — those just hash the whole thing).
    checkpoint_hashes = [
        f"ck_{ck // 1024}k={_hash12(prefix_str[:ck])}"
        for ck in _CACHE_CHECKPOINTS
        if ck < len(prefix_str)
    ]

    prompt = usage_holder.get("input_tokens")
    cached = usage_holder.get("cached_tokens")
    cache_write = usage_holder.get("cache_write_tokens")
    cache_discount = usage_holder.get("cache_discount")
    output = usage_holder.get("output_tokens")
    provider = usage_holder.get("provider")
    parts = [
        "[agex.llm.cache]",
        f"msgs={len(msgs)}",
        f"sys_hash={sys_hash}",
        *checkpoint_hashes,
        f"prefix_hash={prefix_hash}",
        f"prefix_chars={len(prefix_str)}",
        f"prompt_tokens={prompt}",
        f"cached_tokens={cached}",
    ]
    if cache_write is not None:
        parts.append(f"cache_write_tokens={cache_write}")
    if cache_discount is not None:
        parts.append(f"cache_discount={cache_discount}")
    parts.append(f"output_tokens={output}")
    if provider is not None:
        parts.append(f"provider={provider}")
    print(" ".join(parts))


CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}


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
    """Convert generic message dict to OpenAI's format (images as data URIs).

    Preserves all non-content fields (``tool_calls``, ``tool_call_id``,
    ``name``, etc.) — ``translate_messages_to_openai`` sets these when
    the tool-use wire format is in play and they must flow through
    unchanged to the API.

    Cache-control handling:

    - ``content`` is a list of parts → reshape to OpenAI's ``text``/``image_url``
      vocabulary; if ``cache`` is set, add ``cache_control`` to the last block.
    - ``content`` is a string and ``cache`` is set → wrap into a single
      text block carrying ``cache_control``; preserve every other field.
    - ``content`` is ``None`` (e.g. an assistant message whose body is in
      ``tool_calls``) → pass through unchanged.  Skipping the cache marker
      is safer than emitting an invalid ``text: null`` block.
    """
    out = {**message}
    content = message.get("content")

    if isinstance(content, list):
        content_parts = []
        for part in content:
            if not isinstance(part, dict):
                content_parts.append(part)
                continue
            ptype = part.get("type")
            if ptype == "text":
                content_parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image":
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{part.get('image_data', '')}"
                        },
                    }
                )
            else:
                # Already-OpenAI-shape or unknown — pass through.
                content_parts.append(part)
        if cache and content_parts:
            last = dict(content_parts[-1])
            last["cache_control"] = CACHE_CONTROL
            content_parts[-1] = last
        out["content"] = content_parts
        return out

    if cache and isinstance(content, str):
        out["content"] = [
            {"type": "text", "text": content, "cache_control": CACHE_CONTROL}
        ]
        return out

    # content is a string without cache, or None — pass through (with extras).
    return out


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
        use_responses: bool | None = None,
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
        # Endpoint dispatch mirrors the SDK client.  Responses for
        # GPT-5-family / o-series reasoning models, Chat Completions
        # otherwise.  Callers can force either with ``use_responses=``.
        self._use_responses = (
            use_responses if use_responses is not None else _is_reasoning_model(model)
        )
        # Transport seam. When empty api_key is paired with a custom
        # adapter, the adapter is expected to inject auth headers on the
        # way out (e.g., a JS bridge that reads the key from localStorage).
        self._adapter: FetchAdapter = fetch_adapter or DefaultPyfetchAdapter()
        # Reasoning models are the majority now — prefer native
        # thinking (no ``thinking`` / ``report`` schema params) so
        # capable models can't split their call by emitting thinking
        # without code.  Callers routing through OpenRouter to a
        # non-reasoning model can opt out via ``ToolUseWireFormat()``.
        self._wire_format: WireFormat = wire_format or ToolUseWireFormat(
            native_thinking=True
        )

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

        Dispatches to Responses for reasoning-class models (GPT-5 +
        o-series) and Chat Completions for everything else.  Retries
        both paths on network errors (e.g. connection drops on mobile).
        """
        request_kwargs = {**self._kwargs, **kwargs}

        messages_dicts = self._wire_format.render_events(events)
        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        tool_schemas = self._wire_format.tool_schema()

        if self._use_responses:
            url, body, cache_idx = self._build_responses_request(
                system_with_format, messages_dicts, tool_schemas, request_kwargs
            )
            stream_once = self._stream_once_responses
        else:
            url, body, cache_idx = self._build_chat_request(
                system_with_format, messages_dicts, tool_schemas, request_kwargs
            )
            stream_once = self._stream_once_tools

        headers = self._headers()

        for attempt in range(self._STREAM_MAX_RETRIES):
            try:
                async for token in stream_once(url, body, headers, cache_idx):
                    yield token
                return  # success
            except Exception as exc:
                is_network = _is_network_error(exc)
                if not is_network or attempt + 1 >= self._STREAM_MAX_RETRIES:
                    raise
                # Brief pause before retry
                await asyncio.sleep(1)

    def _build_chat_request(
        self,
        system_with_format: str,
        messages_dicts: list[dict],
        tool_schemas: list[dict],
        request_kwargs: dict,
    ) -> tuple[str, dict, int]:
        translated = translate_messages_to_openai(messages_dicts)
        system_msg = _format_message_for_openai(
            {"role": "system", "content": system_with_format}, cache=True
        )
        # Cache breakpoint on the LAST translated message.  In agex's
        # tool-use flow the last message is always a tool result (or a
        # user-text turn-boundary message), and the next turn's request
        # appends *after* it without injecting a fresh user prompt — so
        # caching the last message yields hits on every subsequent turn.
        # Using ``len-2`` (the chat-style default) would land on an
        # assistant-with-tool_calls message whose ``content`` is None,
        # and our cache helper has no content block to attach the marker
        # to → the breakpoint silently disappeared and only the system
        # prompt got cached.
        cache_idx = len(translated) - 1
        conversation = [
            _format_message_for_openai(m, cache=(i == cache_idx))
            for i, m in enumerate(translated)
        ]
        full_messages = [system_msg] + conversation
        body: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "tools": schemas_to_openai_tools(tool_schemas),
            "tool_choice": request_kwargs.pop("tool_choice", "required"),
            "stream": True,
            "stream_options": {"include_usage": True},
            **request_kwargs,
        }
        url = f"{self._base_url}/chat/completions"
        return url, body, cache_idx

    def _build_responses_request(
        self,
        system_with_format: str,
        messages_dicts: list[dict],
        tool_schemas: list[dict],
        request_kwargs: dict,
    ) -> tuple[str, dict, int]:
        """Build the ``/v1/responses`` request body.

        Responses takes ``input`` (a flat list of typed items) rather
        than ``messages``, and ``instructions`` at top level in place
        of a leading system message.  Reasoning round-trip requires
        ``store=False`` + ``include=["reasoning.encrypted_content"]`` —
        the server then embeds the encrypted state on each reasoning
        item, which we capture into :class:`ThinkingEmission` and
        replay verbatim on subsequent turns.
        """
        kwargs = {**request_kwargs}
        # Fold legacy ``reasoning_effort`` into the nested shape.
        if "reasoning_effort" in kwargs and "reasoning" not in kwargs:
            effort = kwargs.pop("reasoning_effort")
            if effort is not None:
                kwargs["reasoning"] = {"effort": effort}
        elif "reasoning_effort" in kwargs:
            kwargs.pop("reasoning_effort")

        input_items = translate_messages_to_openai_responses(messages_dicts)

        kwargs.setdefault("store", False)
        include = list(kwargs.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        kwargs["include"] = include

        body: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            "instructions": system_with_format,
            "tools": schemas_to_openai_responses_tools(tool_schemas),
            "tool_choice": kwargs.pop("tool_choice", "required"),
            "stream": True,
            **kwargs,
        }
        url = f"{self._base_url}/responses"
        # Responses has no message-level cache_control knob we can
        # place a breakpoint on, so the cache-diagnostics log line
        # doesn't apply — pass -1 to skip it.
        return url, body, -1

    async def _stream_once_tools(
        self,
        url: str,
        body: dict,
        headers: dict,
        cache_idx: int = -1,
    ) -> AsyncIterator[TokenChunk]:
        """Single tool-use streaming attempt.

        Reads SSE JSON chunks from the provider, translates them to
        :class:`ToolCallEvent`\\ s, and feeds those through the wire
        format's ``aparse_tool_stream``.

        ``cache_idx`` is the index of the cache_control marker within
        the conversation portion of ``body["messages"]`` (i.e. excluding
        the leading system message); it's used only for logging
        diagnostics about cache prefix placement.
        """
        from agex.llm.sse import parse_sse_events

        response = self._adapter.fetch_stream(url, headers=headers, body=body)

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
        }

        sse_iter = parse_sse_events(response)

        async def chunk_dicts():
            async for payload in sse_iter:
                if not payload.strip():
                    continue
                yield json.loads(payload)

        tool_events = atranslate_openai_stream_to_events(
            chunk_dicts(), usage_holder=usage_holder
        )

        async for token in self._wire_format.aparse_tool_stream(tool_events):
            yield token

        # One log line per request — system + prefix hashes plus
        # actual cache hit numbers, so consecutive turns can be
        # diff'd to identify prefix drift vs. provider-side cache
        # misses.
        try:
            _log_cache_diagnostics(body, cache_idx, usage_holder)
        except Exception:
            pass  # diagnostics must never affect the user's flow

        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

    async def _stream_once_responses(
        self,
        url: str,
        body: dict,
        headers: dict,
        cache_idx: int = -1,  # unused on the Responses path; kept for symmetry
    ) -> AsyncIterator[TokenChunk]:
        """Single Responses-API streaming attempt.

        Reads SSE JSON events from ``/v1/responses``, hands them to
        :func:`atranslate_openai_responses_stream_to_events` for
        translation into :class:`ToolCallEvent`\\ s, and feeds those
        through the wire format's ``aparse_tool_stream`` so the rest
        of the pipeline sees the same provider-agnostic token flow.
        """
        from agex.llm.sse import parse_sse_events

        response = self._adapter.fetch_stream(url, headers=headers, body=body)

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
        }

        sse_iter = parse_sse_events(response)

        async def chunk_dicts():
            async for payload in sse_iter:
                if not payload.strip():
                    continue
                yield json.loads(payload)

        tool_events = atranslate_openai_responses_stream_to_events(
            chunk_dicts(), usage_holder=usage_holder
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
