"""Translate between agex's tool-use wire format and OpenAI's
Responses API (``/v1/responses``) shape.

Required for GPT-5-family reasoning models (``gpt-5``, ``gpt-5-mini``,
``gpt-5.4-nano`` etc.) — the classic Chat Completions endpoint rejects
``reasoning_effort`` in combination with function tools for those
models.  Responses is OpenAI's successor endpoint and is the path
where new features (richer reasoning capture, encrypted round-trip)
live.

Shape differences vs. :mod:`openai_adapter`:

1. **Tool schema** — Responses expects flat ``{"type": "function",
   "name": ..., "description": ..., "parameters": ...}`` items, not
   the nested ``{"type": "function", "function": {...}}`` wrapper
   Chat Completions uses.

2. **Input items** — the ``messages`` list is replaced by ``input``,
   a flat list whose entries are either conversational messages
   (``{"role": "user", "content": [...]}``) or typed items
   (``{"type": "function_call", ...}``, ``{"type": "function_call_output",
   ...}``, ``{"type": "reasoning", "id": ..., "encrypted_content": ...}``).
   Assistant turns fan out into a reasoning item, zero or more
   function_call items, and an optional message item — not a single
   message carrying tool_calls as a sibling field.

3. **Streaming events** — per-item additive/delta/done events
   (``response.output_item.added``, ``response.function_call_arguments.delta``,
   ``response.output_text.delta``, ``response.reasoning_summary_text.delta``,
   ``response.output_item.done``, ``response.completed``).  We walk
   the event stream and translate into the provider-agnostic
   :class:`ToolCallEvent` vocabulary the parser already understands.

4. **Reasoning round-trip** — when ``store=False`` and
   ``include=["reasoning.encrypted_content"]`` are set, reasoning items
   arrive carrying an ``id`` and ``encrypted_content``.  Both must be
   replayed verbatim at the same position on subsequent requests.  We
   pack them into the :class:`ThinkingEmission`'s ``signature`` bytes
   as a small JSON blob so the shared emission model stays
   provider-agnostic — Gemini's raw bytes signature and Anthropic's
   base64 signature still fit the same field without any OpenAI-
   specific schema bleeding through.
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

# --- Schemas ----------------------------------------------------------


def schemas_to_openai_responses_tools(schemas: list[dict]) -> list[dict]:
    """Transform agex's generic tool schemas into Responses ``tools``
    items.

    Responses uses a flat ``{"type": "function", "name": ...,
    "description": ..., "parameters": ...}`` shape — no nested
    ``function`` wrapper like Chat Completions.
    """
    return [
        {
            "type": "function",
            "name": s["name"],
            "description": s["description"],
            "parameters": s["parameters"],
        }
        for s in schemas
    ]


# --- Messages ---------------------------------------------------------


_SIGNATURE_PREFIX = b"openai-responses:"


def encode_reasoning_signature(item_id: str, encrypted_content: str) -> bytes:
    """Pack a Responses reasoning item's ``id`` + ``encrypted_content``
    into the bytes shape :class:`ThinkingEmission.signature` expects.

    Uses a short tag prefix so the decoder can fail fast if a
    non-Responses signature lands here (defensive — in practice the
    openai client only feeds these to the openai responses adapter).
    """
    payload = json.dumps({"id": item_id, "encrypted_content": encrypted_content})
    return _SIGNATURE_PREFIX + payload.encode("utf-8")


def decode_reasoning_signature(sig: bytes) -> tuple[str, str] | None:
    """Inverse of :func:`encode_reasoning_signature`.  Returns
    ``(id, encrypted_content)`` or ``None`` if the bytes aren't a
    Responses-encoded signature.
    """
    if not sig.startswith(_SIGNATURE_PREFIX):
        return None
    try:
        payload = json.loads(sig[len(_SIGNATURE_PREFIX) :].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    item_id = payload.get("id")
    encrypted = payload.get("encrypted_content")
    if not isinstance(item_id, str) or not isinstance(encrypted, str):
        return None
    return item_id, encrypted


def _input_content_from_generic(content: Any) -> Any:
    """Rewrite a user-message ``content`` value to use Responses'
    input-side content-part vocabulary (``input_text`` / ``input_image``).

    Accepts either a plain string (pass-through) or a list of
    ``{"type": ..., ...}`` parts.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    out: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            out.append({"type": "input_text", "text": str(part)})
            continue
        ptype = part.get("type")
        if ptype == "text":
            out.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype == "image":
            out.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{part.get('image_data', '')}",
                }
            )
        else:
            # Unknown shape — pass through; the API will either accept
            # or surface a clear error.
            out.append(part)
    return out


def _tool_result_output_text(content: Any) -> str:
    """Flatten a ``tool_result``'s ``content`` to the plain string
    Responses' ``function_call_output.output`` expects.  Image parts
    become ``[image]`` placeholders — Responses tool outputs are
    single-shot strings, not multimodal blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits: list[str] = []
        for p in content:
            if not isinstance(p, dict):
                bits.append(str(p))
                continue
            ptype = p.get("type")
            if ptype == "text":
                bits.append(p.get("text", ""))
            elif ptype == "image":
                bits.append("[image]")
        return "\n".join(bits)
    return str(content)


def _thinking_block_to_reasoning_item(block: dict) -> dict | None:
    """Unpack a rendered ``thinking`` block back into a Responses
    reasoning input item.

    Requires the signature bytes to decode as a Responses-encoded
    signature (id + encrypted_content).  Returns ``None`` if the
    signature came from a different provider — we silently drop it
    rather than feed the API a malformed reasoning item.
    """
    sig = block.get("signature")
    if not isinstance(sig, bytes):
        return None
    decoded = decode_reasoning_signature(sig)
    if decoded is None:
        return None
    item_id, encrypted_content = decoded
    return {
        "type": "reasoning",
        "id": item_id,
        "encrypted_content": encrypted_content,
        "summary": [],
    }


def translate_messages_to_openai_responses(messages: list[dict]) -> list[dict]:
    """Translate rendered tool-use messages into a Responses ``input``
    list.

    The core difference from Chat Completions: Responses' ``input`` is
    a *flat* list where typed items (``function_call``,
    ``function_call_output``, ``reasoning``) are siblings of
    conversational messages, not nested inside them.  An assistant
    turn that produced a tool call fans out into two or three items;
    tool results become standalone ``function_call_output`` entries.

    Reasoning items only make it back onto the wire if their signature
    decodes as a Responses-encoded blob — signatures from other
    providers (Gemini ``thought_signature``, Anthropic thinking
    signatures) are dropped, since Responses can't use them.
    """
    out: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "tool_use":
                    out.append(
                        {
                            "type": "function_call",
                            "call_id": block["id"],
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input") or {}),
                        }
                    )
                elif btype == "thinking":
                    reasoning = _thinking_block_to_reasoning_item(block)
                    if reasoning is not None:
                        out.append(reasoning)
                elif btype == "text":
                    text = block.get("text", "")
                    if text:
                        out.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text}],
                            }
                        )
            continue

        if role == "user" and isinstance(content, list):
            # Split: tool_results become standalone function_call_output
            # items; remaining parts get rolled into a user message.
            remaining: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "tool_result":
                    out.append(
                        {
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": _tool_result_output_text(
                                block.get("content", "")
                            ),
                        }
                    )
                else:
                    remaining.append(block)
            if remaining:
                out.append(
                    {
                        "role": "user",
                        "content": _input_content_from_generic(remaining),
                    }
                )
            continue

        # Anything else (system/developer messages, plain-string user
        # messages) passes through with its content rewritten for
        # Responses' content-part vocabulary.
        if content is not None:
            out.append(
                {
                    "role": role,
                    "content": _input_content_from_generic(content),
                }
            )

    return out


# --- Streaming --------------------------------------------------------


@dataclass
class _StreamState:
    """Per-stream translator state.

    Tracks open function_call items by ``item_id`` so their arg deltas
    route to the right ``ToolCallStart``.  Reasoning and message items
    are accumulated in-place so their ``done`` events can flush a
    single :class:`ThinkingPart` / :class:`TextPart`.
    """

    # item_id -> call_id (Responses distinguishes the two; our
    # ToolCallEvent vocabulary keys on call_id).
    call_ids: dict[str, str] = field(default_factory=dict)
    # item_id -> accumulated reasoning summary text (or reasoning_text
    # when the model emits full reasoning instead of just summaries).
    reasoning_text: dict[str, list[str]] = field(default_factory=dict)
    # item_id -> accumulated output_text.  Multiple output_text parts
    # on the same message coalesce into one TextPart at item_done.
    message_text: dict[str, list[str]] = field(default_factory=dict)


def _as_dict(chunk: Any) -> dict:
    if isinstance(chunk, dict):
        return chunk
    dump = getattr(chunk, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(chunk)


def _capture_usage(chunk: dict, usage_holder: dict | None) -> None:
    if usage_holder is None:
        return
    if chunk.get("type") != "response.completed":
        return
    response = chunk.get("response") or {}
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return
    if "input_tokens" in usage:
        usage_holder["input_tokens"] = usage["input_tokens"]
    if "output_tokens" in usage:
        usage_holder["output_tokens"] = usage["output_tokens"]
    # Cached-token accounting lives under input_tokens_details.
    details = usage.get("input_tokens_details")
    if isinstance(details, dict) and "cached_tokens" in details:
        usage_holder["cached_tokens"] = details["cached_tokens"]


def _handle_event(state: _StreamState, event: dict) -> Iterator[ToolCallEvent]:
    etype = event.get("type") or ""

    if etype == "response.output_item.added":
        item = event.get("item") or {}
        item_type = item.get("type")
        item_id = item.get("id") or ""
        if item_type == "function_call":
            call_id = item.get("call_id") or item_id or "call_unknown"
            state.call_ids[item_id] = call_id
            yield ToolCallStart(
                call_id=call_id,
                tool_name=item.get("name") or "",
            )
        elif item_type == "reasoning":
            state.reasoning_text.setdefault(item_id, [])
        elif item_type == "message":
            state.message_text.setdefault(item_id, [])
        return

    if etype == "response.function_call_arguments.delta":
        item_id = event.get("item_id") or ""
        call_id = state.call_ids.get(item_id)
        delta = event.get("delta") or ""
        if call_id and delta:
            yield ToolCallArgDelta(call_id=call_id, argument_chunk=delta)
        return

    # Responses emits the same event type for both summary text and
    # full reasoning text; delta-style events carry the piece under
    # ``delta``.  We accumulate everything under the item_id so a
    # single ThinkingPart flushes on item_done.
    if etype in (
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
    ):
        item_id = event.get("item_id") or ""
        delta = event.get("delta") or ""
        if delta:
            state.reasoning_text.setdefault(item_id, []).append(delta)
        return

    if etype == "response.output_text.delta":
        item_id = event.get("item_id") or ""
        delta = event.get("delta") or ""
        if delta:
            state.message_text.setdefault(item_id, []).append(delta)
        return

    if etype == "response.output_item.done":
        item = event.get("item") or {}
        item_type = item.get("type")
        item_id = item.get("id") or ""

        if item_type == "function_call":
            call_id = state.call_ids.pop(item_id, None) or item.get("call_id") or ""
            if call_id:
                yield ToolCallEnd(call_id=call_id)
            return

        if item_type == "reasoning":
            # Prefer the delta-accumulated text; fall back to whatever
            # the final item object carries (covers providers that
            # omit the delta stream and only emit the done event).
            buf = state.reasoning_text.pop(item_id, None)
            text = "".join(buf) if buf else _extract_reasoning_text(item)
            encrypted = item.get("encrypted_content")
            signature: bytes | None = None
            if isinstance(encrypted, str) and encrypted:
                signature = encode_reasoning_signature(item_id, encrypted)
            redacted = bool(encrypted) and not text
            if signature is None and not text:
                return
            yield ThinkingPart(
                signature=signature,
                text=text or None,
                redacted=redacted,
            )
            return

        if item_type == "message":
            buf = state.message_text.pop(item_id, None)
            text = "".join(buf) if buf else _extract_message_text(item)
            if text:
                yield TextPart(text=text)
            return


def _extract_reasoning_text(item: dict) -> str:
    """Fallback for clients that didn't stream delta events — pull
    whatever text hangs off the final reasoning item.  Covers both
    ``summary`` (summary_text parts) and ``content`` (reasoning_text
    parts) shapes.
    """
    bits: list[str] = []
    for part in item.get("summary") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            bits.append(part["text"])
    for part in item.get("content") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            bits.append(part["text"])
    return "\n".join(bits)


def _extract_message_text(item: dict) -> str:
    bits: list[str] = []
    for part in item.get("content") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            bits.append(part["text"])
    return "".join(bits)


def translate_openai_responses_stream_to_events(
    chunks: Iterator[Any],
    usage_holder: dict | None = None,
) -> Iterator[ToolCallEvent]:
    """Translate an OpenAI Responses streaming iterator into
    provider-agnostic :class:`ToolCallEvent`\\ s.

    Accepts either raw event dicts (SSE payloads parsed with
    ``json.loads``) or SDK pydantic-like objects exposing ``model_dump``.
    Usage counts populate ``usage_holder`` when ``response.completed``
    arrives.
    """
    state = _StreamState()
    for chunk in chunks:
        event = _as_dict(chunk)
        _capture_usage(event, usage_holder)
        yield from _handle_event(state, event)


async def atranslate_openai_responses_stream_to_events(
    chunks: AsyncIterator[Any],
    usage_holder: dict | None = None,
) -> AsyncIterator[ToolCallEvent]:
    """Async counterpart to
    :func:`translate_openai_responses_stream_to_events`."""
    state = _StreamState()
    async for chunk in chunks:
        event = _as_dict(chunk)
        _capture_usage(event, usage_holder)
        for ev in _handle_event(state, event):
            yield ev
