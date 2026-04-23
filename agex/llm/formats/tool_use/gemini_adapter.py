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

from .events import ToolCallArgDelta, ToolCallEnd, ToolCallEvent, ToolCallStart

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
                    sig = block.get("signature")
                    if sig is not None:
                        # Gemini 3 requires thought_signature on every
                        # function_call on subsequent turns — without
                        # it the API 400s on the second call.  We
                        # captured this when the stream first delivered
                        # the call; replay it here.
                        fc["thought_signature"] = sig
                    parts.append({"function_call": fc})
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


def _iter_function_call_parts(chunk: Any) -> Iterator[tuple[Any, bytes | None]]:
    """Yield ``(function_call, thought_signature)`` pairs from a chunk.

    Gemini 3 attaches ``thought_signature`` to the ``Part`` that wraps
    the ``function_call`` (bytes on the Part, not on the call itself).
    Walk the candidate content parts to pick them up in lockstep.
    ``chunk.function_calls`` flattens the calls but drops the parts, so
    we work from ``candidates[*].content.parts`` directly.
    """
    candidates = getattr(chunk, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            sig = getattr(part, "thought_signature", None)
            yield fc, sig


def translate_gemini_stream_to_events(
    chunks: Iterator[Any],
    usage_holder: dict | None = None,
) -> Iterator[ToolCallEvent]:
    """Translate a Gemini ``generate_content_stream`` response into
    :class:`ToolCallEvent`\\ s.

    Each chunk is scanned for function_call parts; duplicates across
    chunks (distinguished by id) are deduplicated so a given tool call
    produces exactly one Start/Delta/End triple.  ``thought_signature``
    bytes on each part are forwarded on :class:`ToolCallStart` so the
    wire format can round-trip them on the next turn (Gemini 3
    rejects replayed function_calls that lack their signature).
    """
    seen_ids: set[str] = set()
    counter = [0]
    for chunk in chunks:
        _capture_usage(chunk, usage_holder)
        for fc, sig in _iter_function_call_parts(chunk):
            yield from _emit_function_call(fc, sig, seen_ids, counter)


async def atranslate_gemini_stream_to_events(
    chunks: AsyncIterator[Any],
    usage_holder: dict | None = None,
) -> AsyncIterator[ToolCallEvent]:
    """Async counterpart to :func:`translate_gemini_stream_to_events`."""
    seen_ids: set[str] = set()
    counter = [0]
    async for chunk in chunks:
        _capture_usage(chunk, usage_holder)
        for fc, sig in _iter_function_call_parts(chunk):
            for ev in _emit_function_call(fc, sig, seen_ids, counter):
                yield ev
