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

from .events import ToolCallArgDelta, ToolCallEnd, ToolCallEvent, ToolCallStart

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
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_bits) if text_bits else None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
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
    """Per-stream translator state: tracks open tool calls by index."""

    open_calls: dict[int, str] = field(default_factory=dict)


def _as_dict(chunk: Any) -> dict:
    if isinstance(chunk, dict):
        return chunk
    # SDK pydantic models expose ``model_dump``.
    dump = getattr(chunk, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(chunk)  # best-effort fallback


def _handle_delta(state: _StreamState, delta: dict) -> Iterator[ToolCallEvent]:
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


def _capture_usage(chunk: dict, usage_holder: dict | None) -> None:
    if usage_holder is None:
        return
    usage = chunk.get("usage")
    if not usage:
        return
    usage_holder["input_tokens"] = usage.get("prompt_tokens")
    usage_holder["output_tokens"] = usage.get("completion_tokens")
    # OpenRouter (and OpenAI for cached prompts) reports cache hit size
    # under prompt_tokens_details.cached_tokens.  Surface it so the
    # client can log per-request cache diagnostics.
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if cached is not None:
            usage_holder["cached_tokens"] = cached


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
