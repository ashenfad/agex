"""Translate between agex's tool-use wire format and Anthropic's
Messages API shape.

Three concerns:

1. **Schemas** — Anthropic uses ``{name, description, input_schema}``
   (note the key rename from our generic ``parameters``).  No outer
   ``{"type": "function", "function": ...}`` wrapper like OpenAI.

2. **Messages** — our renderer's output is already very close to
   Anthropic's shape: ``tool_use`` / ``tool_result`` content blocks map
   1:1.  The main delta is image parts, which our renderer emits as
   ``{"type": "image", "image_data": "<b64>"}`` and Anthropic expects
   as ``{"type": "image", "source": {"type": "base64", "media_type":
   "image/png", "data": "<b64>"}}``.

3. **Streaming tool-call events** — Anthropic streams structured SSE
   events (``message_start``, ``content_block_start``,
   ``content_block_delta`` with ``input_json_delta``,
   ``content_block_stop``, ``message_delta``, ``message_stop``).  We
   map them to provider-agnostic :class:`ToolCallEvent`\\ s keyed by
   content-block ``index`` → tool_use ``id``.

Cache-control breakpoints are the client's concern; this module emits
clean messages and a separate helper can inject ``cache_control`` on
whichever blocks the caller chooses.
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from .events import (
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallEvent,
    ToolCallStart,
)

# --- Schemas ----------------------------------------------------------


def schemas_to_anthropic_tools(schemas: list[dict]) -> list[dict]:
    """Rename ``parameters`` → ``input_schema`` (Anthropic's key).
    No outer envelope — Anthropic tools are flat ``{name, description,
    input_schema}``.
    """
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in schemas
    ]


# --- Messages ---------------------------------------------------------


def _translate_content_part(part: dict) -> dict:
    """Translate an image part to Anthropic's ``source`` envelope.
    Text parts and other types pass through unchanged.
    """
    if part.get("type") == "image":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": part.get("image_data", ""),
            },
        }
    return part


def _translate_tool_result(block: dict) -> dict:
    """Translate a ``tool_result`` block.  Its ``content`` may be a
    string (passes through) or a list of content parts (each part
    translated).
    """
    inner = block.get("content")
    translated_inner: Any = inner
    if isinstance(inner, list):
        translated_inner = [_translate_content_part(p) for p in inner]
    return {
        "type": "tool_result",
        "tool_use_id": block["tool_use_id"],
        "content": translated_inner,
    }


def _translate_thinking_block(block: dict) -> dict:
    """Round-trip a signed thinking block back into Anthropic's native
    content-block shape.

    Non-redacted: ``{"type": "thinking", "thinking": <text>, "signature":
    <str>}``.  Redacted: ``{"type": "redacted_thinking", "data":
    <str>}``.  The signature field stores bytes in the emission model
    for cross-provider consistency (Gemini uses bytes); we decode back
    to the Anthropic-native string at the boundary.
    """
    sig = block.get("signature")
    if isinstance(sig, bytes):
        sig_str = sig.decode("utf-8", errors="replace")
    else:
        sig_str = sig or ""
    if block.get("redacted"):
        return {"type": "redacted_thinking", "data": sig_str}
    return {
        "type": "thinking",
        "thinking": block.get("text", "") or "",
        "signature": sig_str,
    }


def translate_messages_to_anthropic(messages: list[dict]) -> list[dict]:
    """Translate tool-use rendered messages to Anthropic's Messages
    API shape.

    Our renderer already emits ``tool_use`` / ``tool_result`` blocks
    matching Anthropic's vocabulary; the only per-block work is
    translating image parts to Anthropic's ``source`` envelope and
    converting our generic ``thinking`` blocks back into the
    ``thinking`` / ``redacted_thinking`` content blocks Anthropic
    expects.  Messages with string ``content`` pass through unchanged.
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_content: list[dict] = []
        for block in content:
            btype = block.get("type")
            if btype == "image":
                new_content.append(_translate_content_part(block))
            elif btype == "tool_result":
                new_content.append(_translate_tool_result(block))
            elif btype == "thinking":
                new_content.append(_translate_thinking_block(block))
            else:
                new_content.append(block)
        out.append({"role": msg["role"], "content": new_content})
    return out


def apply_cache_control(
    messages: list[dict], cache_index: int, ttl: str = "1h"
) -> list[dict]:
    """Return a copy of ``messages`` with a ``cache_control`` breakpoint
    on the last content block of ``messages[cache_index]``.

    Out-of-range indices are silently ignored.  The last content block
    is chosen because Anthropic applies caching to the prefix ending at
    the marked block.
    """
    if not messages or cache_index < 0 or cache_index >= len(messages):
        return messages
    out = [dict(m) for m in messages]
    target = out[cache_index]
    content = target.get("content")
    cc = {"type": "ephemeral", "ttl": ttl}
    if isinstance(content, list) and content:
        new_blocks = [dict(b) for b in content]
        new_blocks[-1] = {**new_blocks[-1], "cache_control": cc}
        target["content"] = new_blocks
    elif isinstance(content, str):
        # Promote string content to a single text block with cache_control.
        target["content"] = [{"type": "text", "text": content, "cache_control": cc}]
    return out


# --- Streaming --------------------------------------------------------


@dataclass
class _ThinkingState:
    """Accumulator for a streaming Claude thinking block.

    Claude streams thinking blocks as ``content_block_start`` → many
    ``thinking_delta`` / ``signature_delta`` events → ``content_block_stop``.
    We collect the text and signature pieces here and emit a single
    :class:`ThinkingPart` once the block closes.

    For ``redacted_thinking`` blocks the encrypted payload arrives on
    ``content_block_start.data`` (not via deltas) — we stash it here
    and preserve it as the emission's ``signature`` bytes with
    ``redacted=True`` so the renderer can replay it as
    ``{"type": "redacted_thinking", "data": ...}`` verbatim.
    """

    text: str = ""
    signature: str = ""
    redacted: bool = False
    data: str = ""


@dataclass
class _StreamState:
    """Per-stream translator state.  Tracks open tool-use blocks by
    content-block ``index`` → ``tool_use.id`` and open thinking
    blocks by content-block ``index`` → :class:`_ThinkingState`.
    """

    open_by_index: dict[int, str] = field(default_factory=dict)
    thinking_by_index: dict[int, _ThinkingState] = field(default_factory=dict)


def _as_dict(event: Any) -> dict:
    if isinstance(event, dict):
        return event
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(event)  # best-effort fallback


def _total_input(usage: dict) -> int | None:
    """Sum regular + cache-creation + cache-read input tokens.

    Anthropic splits input tokens across three buckets when prompt
    caching is in play; callers want the total.
    """
    if "input_tokens" not in usage and not any(
        k in usage for k in ("cache_creation_input_tokens", "cache_read_input_tokens")
    ):
        return None
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
    )


def _capture_usage(event_dict: dict, usage_holder: dict | None) -> None:
    if usage_holder is None:
        return
    etype = event_dict.get("type")
    if etype == "message_start":
        msg = event_dict.get("message") or {}
        usage = msg.get("usage") or {}
        total_in = _total_input(usage)
        if total_in is not None:
            usage_holder["input_tokens"] = total_in
        if "output_tokens" in usage:
            usage_holder["output_tokens"] = usage["output_tokens"]
    elif etype == "message_delta":
        usage = event_dict.get("usage") or {}
        total_in = _total_input(usage)
        if total_in is not None:
            usage_holder["input_tokens"] = total_in
        if "output_tokens" in usage:
            usage_holder["output_tokens"] = usage["output_tokens"]


def _handle_event(state: _StreamState, event_dict: dict) -> Iterator[ToolCallEvent]:
    etype = event_dict.get("type")
    if etype == "content_block_start":
        idx = event_dict.get("index")
        if idx is None:
            return
        block = event_dict.get("content_block") or {}
        btype = block.get("type")
        if btype == "tool_use":
            call_id = block.get("id") or f"call_{idx}"
            name = block.get("name") or ""
            state.open_by_index[idx] = call_id
            yield ToolCallStart(call_id=call_id, tool_name=name)
        elif btype == "thinking":
            state.thinking_by_index[idx] = _ThinkingState()
        elif btype == "redacted_thinking":
            # ``data`` on redacted blocks is the opaque encrypted
            # payload — not delta-streamed, delivered whole on
            # block_start.
            state.thinking_by_index[idx] = _ThinkingState(
                redacted=True, data=block.get("data", "") or ""
            )
    elif etype == "content_block_delta":
        idx = event_dict.get("index")
        delta = event_dict.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "input_json_delta" and idx in state.open_by_index:
            partial = delta.get("partial_json") or ""
            if partial:
                yield ToolCallArgDelta(
                    call_id=state.open_by_index[idx],
                    argument_chunk=partial,
                )
        elif dtype == "thinking_delta" and idx in state.thinking_by_index:
            state.thinking_by_index[idx].text += delta.get("thinking", "") or ""
        elif dtype == "signature_delta" and idx in state.thinking_by_index:
            state.thinking_by_index[idx].signature += delta.get("signature", "") or ""
    elif etype == "content_block_stop":
        idx = event_dict.get("index")
        if idx is None:
            return
        call_id = state.open_by_index.pop(idx, None)
        if call_id:
            yield ToolCallEnd(call_id=call_id)
            return
        thinking = state.thinking_by_index.pop(idx, None)
        if thinking is not None:
            yield from _emit_thinking_part(thinking)


def _emit_thinking_part(thinking: _ThinkingState) -> Iterator[ToolCallEvent]:
    """Yield a :class:`ThinkingPart` for an accumulated thinking block.

    For regular ``thinking`` blocks the signature ships as an opaque
    string (Anthropic's API shape); we encode it to bytes so the
    emission's ``signature: bytes | None`` contract holds across
    providers.  For ``redacted_thinking`` blocks the signed payload
    lives in ``data`` instead — we stash it in the same ``signature``
    field with ``redacted=True`` so the renderer can round-trip it
    back as ``{"type": "redacted_thinking", "data": ...}``.

    Nothing is emitted for blocks that ended up with neither text nor
    signature nor redacted payload (shouldn't happen, but belt-and-
    suspenders).
    """
    if thinking.redacted:
        sig_bytes = thinking.data.encode("utf-8") if thinking.data else None
        if sig_bytes is None:
            return
        yield ThinkingPart(signature=sig_bytes, text=None, redacted=True)
        return
    sig_bytes = thinking.signature.encode("utf-8") if thinking.signature else None
    text = thinking.text or None
    if sig_bytes is None and not text:
        return
    yield ThinkingPart(signature=sig_bytes, text=text, redacted=False)


def translate_anthropic_stream_to_events(
    events: Iterator[Any],
    usage_holder: dict | None = None,
) -> Iterator[ToolCallEvent]:
    """Translate an Anthropic Messages-API event stream into
    :class:`ToolCallEvent`\\ s.

    Accepts either raw dicts (SSE ``data`` payloads already parsed) or
    objects exposing ``model_dump`` (SDK ``MessageStreamEvent``).  Any
    tool-use blocks still open when the stream ends emit a safety-net
    ``ToolCallEnd`` so downstream consumers always see a clean close.
    """
    state = _StreamState()
    for event in events:
        event_dict = _as_dict(event)
        _capture_usage(event_dict, usage_holder)
        yield from _handle_event(state, event_dict)
    for call_id in state.open_by_index.values():
        yield ToolCallEnd(call_id=call_id)
    state.open_by_index.clear()
    for thinking in state.thinking_by_index.values():
        yield from _emit_thinking_part(thinking)
    state.thinking_by_index.clear()


async def atranslate_anthropic_stream_to_events(
    events: AsyncIterator[Any],
    usage_holder: dict | None = None,
) -> AsyncIterator[ToolCallEvent]:
    """Async counterpart to
    :func:`translate_anthropic_stream_to_events`."""
    state = _StreamState()
    async for event in events:
        event_dict = _as_dict(event)
        _capture_usage(event_dict, usage_holder)
        for ev in _handle_event(state, event_dict):
            yield ev
    for call_id in state.open_by_index.values():
        yield ToolCallEnd(call_id=call_id)
    state.open_by_index.clear()
    for thinking in state.thinking_by_index.values():
        for ev in _emit_thinking_part(thinking):
            yield ev
    state.thinking_by_index.clear()
