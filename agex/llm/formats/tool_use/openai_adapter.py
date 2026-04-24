"""Translate between agex's tool-use wire format and OpenAI's Chat
Completions shape.

Three concerns:

1. **Schemas** — OpenAI expects ``tools=[{"type": "function", "function":
   {...}}]``.  Our schemas are the inner ``{...}``; wrap them.

2. **Messages** — our renderer produces generic dicts whose ``content``
   is a list of ``tool_use`` / ``tool_result`` blocks.  OpenAI encodes
   the same information via:

   - Assistant messages with ``tool_calls: [{id, type, function}]``.
   - Separate ``role: "tool"`` messages keyed by ``tool_call_id``.

3. **Streaming tool-call events** — OpenAI streams partial tool calls
   via ``choices[0].delta.tool_calls``.  We translate the delta stream
   into provider-agnostic :class:`ToolCallEvent`\\ s that the wire
   format's ``parse_tool_stream`` consumes.

All translators are stateless over message/schema lists; the streaming
translators keep per-``index`` state internally.
"""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from .events import (
    TextPart,
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallEvent,
    ToolCallStart,
)

# OpenRouter's unified reasoning-tokens API carries a
# ``reasoning_details`` array on assistant messages — one entry per
# reasoning block, typed (``reasoning.summary`` / ``reasoning.text`` /
# ``reasoning.encrypted``) and format-tagged (``anthropic-claude-v1``,
# ``openai-responses-v1``, ``google-gemini-v1``, ...).  The docs
# require "the entire sequence of consecutive reasoning blocks" to
# match the original response on replay, so we pack the full array
# verbatim into :class:`ThinkingEmission.signature` bytes with a
# short tag prefix — same pattern the Responses adapter uses.  That
# keeps the emission model provider-agnostic while still round-
# tripping every opaque byte OpenRouter expects.
_OPENROUTER_REASONING_PREFIX = b"openrouter-reasoning:"


def encode_openrouter_reasoning(details: list) -> bytes:
    """Pack OpenRouter's ``reasoning_details`` array into signature
    bytes so it can ride on :class:`ThinkingEmission.signature` and
    round-trip on the next turn.
    """
    return _OPENROUTER_REASONING_PREFIX + json.dumps(details).encode("utf-8")


def decode_openrouter_reasoning(sig: Any) -> list | None:
    """Inverse of :func:`encode_openrouter_reasoning`.  Returns the
    array or ``None`` if the bytes aren't an OpenRouter-encoded
    signature (defensive — keeps Gemini/Anthropic/Responses
    signatures from being mis-replayed here).
    """
    if not isinstance(sig, bytes) or not sig.startswith(_OPENROUTER_REASONING_PREFIX):
        return None
    try:
        payload = json.loads(sig[len(_OPENROUTER_REASONING_PREFIX) :].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    return payload


# --- Schemas ----------------------------------------------------------


def schemas_to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Wrap agex's generic tool schemas in OpenAI's ``tools`` shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in schemas
    ]


# --- Messages ---------------------------------------------------------


def _stringify_tool_result_content(content: Any) -> str:
    """Flatten a tool_result's ``content`` to a plain string.

    OpenAI's Chat Completions ``role: "tool"`` messages accept only
    string content.  Image parts become ``[image]`` placeholders; a
    subsequent user message is expected to carry the actual image if
    the caller needs multimodal observation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits: list[str] = []
        for p in content:
            if not isinstance(p, dict):
                bits.append(str(p))
                continue
            if p.get("type") == "text":
                bits.append(p.get("text", ""))
            elif p.get("type") == "image":
                bits.append("[image]")
        return "\n".join(bits)
    return str(content)


def translate_messages_to_openai(messages: list[dict]) -> list[dict]:
    """Translate tool-use rendered messages to OpenAI Chat Completions
    message dicts.

    Rules:

    - Assistant messages with ``tool_use`` blocks become
      ``{"role": "assistant", "content": None, "tool_calls": [...]}``
      with JSON-serialized arguments.
    - User messages with ``tool_result`` blocks become one
      ``{"role": "tool", ...}`` message per result; any remaining text
      parts become a trailing ``{"role": "user", ...}`` message.
    - Plain text user messages pass through with flattened content.
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            tool_calls: list[dict] = []
            text_bits: list[str] = []
            reasoning_details: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
                elif btype == "text":
                    text_bits.append(block.get("text", ""))
                elif btype == "thinking":
                    # Unpack an OpenRouter reasoning_details array
                    # back onto the assistant message.  OpenRouter
                    # requires the full sequence of reasoning blocks
                    # to round-trip verbatim; we only emit the field
                    # when the signature decodes as OpenRouter-shaped
                    # (signatures from Gemini / Anthropic / OpenAI
                    # Responses are silently ignored — those providers
                    # don't go through this adapter on replay).
                    decoded = decode_openrouter_reasoning(block.get("signature"))
                    if decoded is not None:
                        reasoning_details.extend(decoded)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_bits) if text_bits else None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            if reasoning_details:
                assistant_msg["reasoning_details"] = reasoning_details
            out.append(assistant_msg)
            continue

        if role == "user" and isinstance(content, list):
            tool_results: list[dict] = []
            text_or_image_parts: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "tool_result":
                    tool_results.append(block)
                elif btype in ("text", "image"):
                    text_or_image_parts.append(block)
            for tr in tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": _stringify_tool_result_content(
                            tr.get("content", "")
                        ),
                    }
                )
            if text_or_image_parts:
                if all(p.get("type") == "text" for p in text_or_image_parts):
                    out.append(
                        {
                            "role": "user",
                            "content": "".join(
                                p.get("text", "") for p in text_or_image_parts
                            ),
                        }
                    )
                else:
                    # Mixed text + image — keep as list for multimodal.
                    out.append({"role": "user", "content": text_or_image_parts})
            continue

        # Plain string or other roles — pass through unchanged.
        out.append(msg)
    return out


# --- Streaming --------------------------------------------------------


@dataclass
class _StreamState:
    """Per-stream translator state: tracks open tool calls by index,
    accumulates any plain-text ``delta.content`` chunks so they can
    be flushed as a :class:`TextPart` at stream end, and buffers any
    OpenRouter ``delta.reasoning_details`` entries so they can be
    flushed as a single :class:`ThinkingPart` carrying the full
    ``reasoning_details`` array encoded into its signature.
    """

    open_calls: dict[int, str] = field(default_factory=dict)
    text_buf: list[str] = field(default_factory=list)
    # Keyed by ``index`` within the assistant turn.  Each entry is
    # a dict mirroring the reasoning_details shape OpenRouter sends
    # (``type``, ``format``, ``id``, ``index`` + type-specific
    # content fields like ``text`` / ``summary`` / ``data``).  Text
    # fields concatenate across deltas; non-text fields take the
    # latest non-empty value so a late-arriving chunk with an ``id``
    # or ``format`` fills in whatever was missing earlier.
    reasoning_by_index: dict[int, dict] = field(default_factory=dict)


def _as_dict(chunk: Any) -> dict:
    if isinstance(chunk, dict):
        return chunk
    # SDK pydantic models expose ``model_dump``.
    dump = getattr(chunk, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(chunk)  # best-effort fallback


def _handle_delta(state: _StreamState, delta: dict) -> Iterator[ToolCallEvent]:
    # Plain assistant text arrives as ``delta.content`` string chunks
    # when the model mixes prose with tool calls (or replies with just
    # prose).  Buffer and flush as a single TextPart at stream end so
    # we don't emit many tiny fragments.
    content = delta.get("content")
    if isinstance(content, str) and content:
        state.text_buf.append(content)

    # OpenRouter's unified reasoning tokens arrive as
    # ``delta.reasoning_details`` — accumulate by ``index`` so the
    # final ThinkingPart carries the same array shape the server
    # sent.  Multiple deltas for the same index concatenate text
    # fields; other fields (format, type, id, data) take the latest
    # non-empty value.
    for rd in delta.get("reasoning_details") or []:
        if not isinstance(rd, dict):
            continue
        idx = rd.get("index")
        if idx is None:
            continue
        slot = state.reasoning_by_index.setdefault(idx, {})
        for key, val in rd.items():
            if key in ("text", "summary") and isinstance(val, str):
                slot[key] = (slot.get(key) or "") + val
            elif val is not None and val != "":
                slot[key] = val

    tool_calls = delta.get("tool_calls") or []
    for tc in tool_calls:
        idx = tc.get("index")
        if idx is None:
            continue
        if idx not in state.open_calls:
            call_id = tc.get("id") or f"call_{idx}"
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            state.open_calls[idx] = call_id
            yield ToolCallStart(call_id=call_id, tool_name=name)
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if args:
            yield ToolCallArgDelta(
                call_id=state.open_calls[idx],
                argument_chunk=args,
            )


def _close_open(state: _StreamState) -> Iterator[ToolCallEvent]:
    for call_id in state.open_calls.values():
        yield ToolCallEnd(call_id=call_id)
    state.open_calls.clear()
    if state.text_buf:
        text = "".join(state.text_buf)
        state.text_buf.clear()
        if text:
            yield TextPart(text=text)
    if state.reasoning_by_index:
        # Flush all reasoning blocks as a single ThinkingPart — the
        # whole array rides in the signature so the next turn can
        # replay it verbatim.  Surfaceable text (``summary`` or
        # ``text`` entries) aggregates into the visible text; opaque
        # encrypted blocks render as redacted.
        details = [
            state.reasoning_by_index[k] for k in sorted(state.reasoning_by_index.keys())
        ]
        state.reasoning_by_index.clear()
        text_bits: list[str] = []
        for d in details:
            dtype = d.get("type") or ""
            if dtype.endswith("summary") or dtype.endswith("text"):
                t = d.get("text") or d.get("summary")
                if t:
                    text_bits.append(t)
        surfaced = "\n".join(text_bits) if text_bits else None
        yield ThinkingPart(
            signature=encode_openrouter_reasoning(details),
            text=surfaced,
            redacted=surfaced is None,
        )


def _capture_usage(chunk: dict, usage_holder: dict | None) -> None:
    if usage_holder is None:
        return

    # Provider identity travels at the top of each OpenRouter chunk
    # (usually only the first or last carries it).  Capture whenever
    # we see it so the client can log which upstream actually served
    # the request — sticky-routing diagnostics.
    provider = chunk.get("provider")
    if provider is not None:
        usage_holder["provider"] = provider

    usage = chunk.get("usage")
    if not usage:
        return
    usage_holder["input_tokens"] = usage.get("prompt_tokens")
    usage_holder["output_tokens"] = usage.get("completion_tokens")

    # OpenRouter / OpenAI report cache hit size under
    # prompt_tokens_details.cached_tokens; some providers also surface
    # cache_write_tokens (tokens written to cache on this request).
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if cached is not None:
            usage_holder["cached_tokens"] = cached
        cache_write = details.get("cache_write_tokens")
        if cache_write is not None:
            usage_holder["cache_write_tokens"] = cache_write

    # Top-level cache_discount (cost savings) is a useful at-a-glance
    # signal that caching actually happened, especially when token
    # counts alone don't tell the whole story.
    discount = chunk.get("cache_discount")
    if discount is not None:
        usage_holder["cache_discount"] = discount


def translate_openai_stream_to_events(
    chunks: Iterator[Any],
    usage_holder: dict | None = None,
) -> Iterator[ToolCallEvent]:
    """Translate OpenAI Chat Completion stream chunks into
    :class:`ToolCallEvent`\\ s.

    Accepts either raw dicts (SSE) or objects exposing ``model_dump``
    (SDK pydantic models).  If ``usage_holder`` is provided, the
    ``input_tokens`` / ``output_tokens`` keys are populated when a
    usage-bearing chunk arrives (final chunk when
    ``stream_options.include_usage=True``).

    Closes open tool calls on the first chunk whose
    ``choices[0].finish_reason`` is non-null, then again at stream end
    as a safety net.
    """
    state = _StreamState()
    finished = False
    for chunk in chunks:
        chunk_dict = _as_dict(chunk)
        _capture_usage(chunk_dict, usage_holder)
        choices = chunk_dict.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            yield from _handle_delta(state, delta)
            if not finished and choices[0].get("finish_reason"):
                yield from _close_open(state)
                finished = True
    if not finished:
        yield from _close_open(state)


async def atranslate_openai_stream_to_events(
    chunks: AsyncIterator[Any],
    usage_holder: dict | None = None,
) -> AsyncIterator[ToolCallEvent]:
    """Async counterpart to :func:`translate_openai_stream_to_events`."""
    state = _StreamState()
    finished = False
    async for chunk in chunks:
        chunk_dict = _as_dict(chunk)
        _capture_usage(chunk_dict, usage_holder)
        choices = chunk_dict.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            for ev in _handle_delta(state, delta):
                yield ev
            if not finished and choices[0].get("finish_reason"):
                for ev in _close_open(state):
                    yield ev
                finished = True
    if not finished:
        for ev in _close_open(state):
            yield ev
