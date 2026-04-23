"""Translate between agex's tool-use wire format and Gemini's
``generate_content`` shape (``google-genai`` SDK).

Three concerns:

1. **Schemas** — Gemini expects ``Tool(function_declarations=[{name,
   description, parameters}, ...])`` — same inner shape as OpenAI, no
   ``type: "function"`` wrapper.  The client is responsible for
   wrapping the declarations in a ``Tool`` alongside any grounding
   tools (``google_search``, ``url_context``).

2. **Messages** — Gemini uses ``Content(role="user"|"model",
   parts=[...])``.  Function calls arrive as ``Part(function_call={id,
   name, args})`` in a model turn; function responses as
   ``Part(function_response={id, name, response})`` in a user turn.

   Critically, ``FunctionResponse`` is keyed by **function name**
   (plus optional ``id``).  Our renderer's ``tool_result`` blocks only
   carry ``tool_use_id``, so the translator walks messages in order
   and maintains an ``id → name`` map populated from preceding
   ``tool_use`` blocks, recovering the name when it hits the matching
   ``tool_result``.

3. **Streaming** — Gemini function-call parts arrive essentially
   complete (``fc.args`` is a full dict, not a delta stream).  We emit
   ``ToolCallStart`` + a single ``ToolCallArgDelta`` containing the
   JSON-serialized args + ``ToolCallEnd`` in one burst per unique call
   id, so the existing tool-use parser can run it through
   ``JsonStringExtractor`` unchanged.
"""

import json
from typing import Any, AsyncIterator, Iterator

from .events import (
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallEvent,
    ToolCallStart,
)

# --- Schemas ----------------------------------------------------------


def schemas_to_gemini_function_declarations(schemas: list[dict]) -> list[dict]:
    """Return a list of ``FunctionDeclaration`` dicts suitable for
    wrapping in ``Tool(function_declarations=...)``.

    Our generic schemas already use ``parameters`` — Gemini's name —
    so this is a pure name/description/parameters projection.
    """
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["parameters"],
        }
        for s in schemas
    ]


# --- Messages ---------------------------------------------------------


def _image_part_to_gemini(part: dict) -> dict:
    return {
        "inline_data": {
            "mime_type": "image/png",
            "data": part.get("image_data", ""),
        }
    }


def _tool_result_response_payload(content: Any) -> dict:
    """Wrap a tool_result's content in the ``{"result": ...}`` dict
    that Gemini's ``FunctionResponse.response`` expects.

    Images are flattened to text placeholders — Gemini's function
    response payload is a plain JSON object, not a multimodal
    content-part list.  Callers that need images post-tool should send
    them separately as user-turn parts.
    """
    if isinstance(content, str):
        return {"result": content}
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
        return {"result": "\n".join(bits)}
    return {"result": str(content)}


def translate_messages_to_gemini(messages: list[dict]) -> list[dict]:
    """Translate tool-use rendered messages to Gemini ``Content`` dicts.

    Returns a list of ``{"role": "user"|"model", "parts": [...]}``
    dicts the SDK validates into ``types.Content``.

    Empty-part messages are dropped — Gemini rejects content with no
    parts.
    """
    id_to_name: dict[str, str] = {}
    out: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        gemini_role = "model" if role == "assistant" else "user"
        content = msg.get("content")

        parts: list[dict] = []

        if isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append({"text": text})
                elif btype == "thinking":
                    # Round-trip a signed thought part.  Gemini's rules
                    # require signatures returned in thought parts to
                    # be returned in thought parts; the Part carries
                    # ``thought=True``, ``thought_signature`` bytes,
                    # and optionally the surfaced text.
                    thought_part: dict[str, Any] = {"thought": True}
                    thought_text = block.get("text")
                    if thought_text:
                        thought_part["text"] = thought_text
                    thought_sig = block.get("signature")
                    if thought_sig is not None:
                        thought_part["thought_signature"] = thought_sig
                    parts.append(thought_part)
                elif btype == "image":
                    parts.append(_image_part_to_gemini(block))
                elif btype == "tool_use":
                    tool_id = block["id"]
                    name = block["name"]
                    id_to_name[tool_id] = name
                    fc: dict[str, Any] = {
                        "id": tool_id,
                        "name": name,
                        "args": block.get("input") or {},
                    }
                    part: dict[str, Any] = {"function_call": fc}
                    sig = block.get("signature")
                    if sig is not None:
                        # Gemini 3 requires thought_signature on every
                        # function_call on subsequent turns — without
                        # it the API 400s on the second call.  The
                        # signature lives on the wrapping ``Part`` as a
                        # sibling of ``function_call`` (NOT inside the
                        # FunctionCall object itself — the SDK's
                        # pydantic validator rejects the extra field).
                        part["thought_signature"] = sig
                    parts.append(part)
                elif btype == "tool_result":
                    tool_id = block["tool_use_id"]
                    name = id_to_name.get(tool_id, "")
                    parts.append(
                        {
                            "function_response": {
                                "id": tool_id,
                                "name": name,
                                "response": _tool_result_response_payload(
                                    block.get("content")
                                ),
                            }
                        }
                    )
                # Unknown block types silently skipped.
        elif isinstance(content, str):
            if content:
                parts.append({"text": content})

        if parts:
            out.append({"role": gemini_role, "parts": parts})

    return out


# --- Streaming --------------------------------------------------------


def _capture_usage(chunk: Any, usage_holder: dict | None) -> None:
    if usage_holder is None:
        return
    um = getattr(chunk, "usage_metadata", None)
    if um is None:
        return
    # prompt_token_count + candidates_token_count — Gemini's accounting.
    ptc = getattr(um, "prompt_token_count", None)
    ctc = getattr(um, "candidates_token_count", None)
    if ptc is not None:
        usage_holder["input_tokens"] = ptc
    if ctc is not None:
        usage_holder["output_tokens"] = ctc


def _call_id_for(fc: Any, counter: int) -> str:
    """Use the SDK-provided ``id`` if set; otherwise synthesize one
    stable across a single stream."""
    provided = getattr(fc, "id", None)
    if provided:
        return provided
    name = getattr(fc, "name", None) or "fn"
    return f"call_{counter}_{name}"


def _emit_function_call(
    fc: Any,
    signature: bytes | None,
    seen_ids: set[str],
    counter_cell: list[int],
) -> Iterator[ToolCallEvent]:
    call_id = _call_id_for(fc, counter_cell[0])
    if call_id in seen_ids:
        # Gemini sometimes re-yields the same call across chunks; ignore
        # once we've already emitted Start/Delta/End for it.
        return
    seen_ids.add(call_id)
    counter_cell[0] += 1
    name = getattr(fc, "name", None) or ""
    args = getattr(fc, "args", None) or {}
    yield ToolCallStart(call_id=call_id, tool_name=name, signature=signature)
    yield ToolCallArgDelta(call_id=call_id, argument_chunk=json.dumps(args))
    yield ToolCallEnd(call_id=call_id)


def _iter_stream_parts(
    chunk: Any,
) -> Iterator[tuple[str, Any, bytes | None, str | None, bool]]:
    """Walk Gemini candidate parts in order, yielding one tuple per
    relevant part.

    Kind is ``"function_call"`` for parts carrying a ``function_call``
    (returns the function_call + signature + None + False).  Kind is
    ``"thought"`` for parts that carry a ``thought_signature``
    (regardless of whether they also have ``thought=True`` or text
    content) but no function_call — Gemini 3 uses these to sign
    reasoning that preceded subsequent function calls, and the
    signatures must round-trip at the same position on replay.

    ``chunk.function_calls`` flattens calls but drops both their
    Part-level signature and any sibling thought parts, so we work
    from ``candidates[*].content.parts`` directly.
    """
    candidates = getattr(chunk, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            sig = getattr(part, "thought_signature", None)
            if fc is not None:
                yield ("function_call", fc, sig, None, False)
                continue
            is_thought = bool(getattr(part, "thought", False))
            text = getattr(part, "text", None) or None
            if sig is not None or is_thought:
                yield ("thought", None, sig, text, is_thought)


def _maybe_emit_thinking(
    sig: bytes | None, text: str | None, is_thought: bool
) -> Iterator[ToolCallEvent]:
    """Emit a :class:`ThinkingPart` if the part carries anything worth
    preserving.  Purely empty thought parts (no signature, no text)
    are dropped — nothing to round-trip.
    """
    if sig is None and not text:
        return
    yield ThinkingPart(signature=sig, text=text, redacted=bool(is_thought and not text))


def translate_gemini_stream_to_events(
    chunks: Iterator[Any],
    usage_holder: dict | None = None,
) -> Iterator[ToolCallEvent]:
    """Translate a Gemini ``generate_content_stream`` response into
    :class:`ToolCallEvent`\\ s.

    Each chunk's candidate parts are walked in order.  Function-call
    parts produce a Start/ArgDelta/End triple; signed thought parts
    produce a :class:`ThinkingPart` that the parser materializes into
    a :class:`ThinkingEmission`.  Gemini 3 rejects subsequent turns
    that don't replay signatures at the same position, so preserving
    thought parts is load-bearing.
    """
    seen_ids: set[str] = set()
    counter = [0]
    for chunk in chunks:
        _capture_usage(chunk, usage_holder)
        for kind, body, sig, text, is_thought in _iter_stream_parts(chunk):
            if kind == "function_call":
                yield from _emit_function_call(body, sig, seen_ids, counter)
            else:  # "thought"
                yield from _maybe_emit_thinking(sig, text, is_thought)


async def atranslate_gemini_stream_to_events(
    chunks: AsyncIterator[Any],
    usage_holder: dict | None = None,
) -> AsyncIterator[ToolCallEvent]:
    """Async counterpart to :func:`translate_gemini_stream_to_events`."""
    seen_ids: set[str] = set()
    counter = [0]
    async for chunk in chunks:
        _capture_usage(chunk, usage_holder)
        for kind, body, sig, text, is_thought in _iter_stream_parts(chunk):
            if kind == "function_call":
                for ev in _emit_function_call(body, sig, seen_ids, counter):
                    yield ev
            else:  # "thought"
                for ev in _maybe_emit_thinking(sig, text, is_thought):
                    yield ev
