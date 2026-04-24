"""Tests for the OpenAI Responses-API adapter in
agex.llm.formats.tool_use.openai_responses_adapter.
"""

import json

import pytest

from agex.llm.formats.tool_use import (
    TextPart,
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallStart,
)
from agex.llm.formats.tool_use.openai_responses_adapter import (
    atranslate_openai_responses_stream_to_events,
    decode_reasoning_signature,
    encode_reasoning_signature,
    schemas_to_openai_responses_tools,
    translate_messages_to_openai_responses,
    translate_openai_responses_stream_to_events,
)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class TestSchemasToResponsesTools:
    def test_flat_shape_no_function_wrapper(self):
        """Responses uses a flat ``{"type": "function", "name": ...}``
        shape, not the ``{"type": "function", "function": {...}}``
        wrapper Chat Completions uses."""
        schemas = [
            {
                "name": "python_action",
                "description": "Run Python.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        out = schemas_to_openai_responses_tools(schemas)
        assert out == [
            {
                "type": "function",
                "name": "python_action",
                "description": "Run Python.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        # No nested ``function`` wrapper.
        assert "function" not in out[0]


# ---------------------------------------------------------------------------
# Signature encoding
# ---------------------------------------------------------------------------


class TestReasoningSignatureCodec:
    def test_round_trip(self):
        sig = encode_reasoning_signature("rs_123", "opaque-enc-content")
        assert decode_reasoning_signature(sig) == ("rs_123", "opaque-enc-content")

    def test_non_responses_bytes_decode_to_none(self):
        """Gemini-shaped raw signature bytes don't have our tag prefix
        and must fail-closed so the translator drops them instead of
        feeding the API garbage."""
        assert decode_reasoning_signature(b"\x01\x02\x03") is None


# ---------------------------------------------------------------------------
# Message / input-item translation
# ---------------------------------------------------------------------------


class TestTranslateMessagesToResponses:
    def test_user_text_becomes_input_text_part(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        out = translate_messages_to_openai_responses(msgs)
        assert out == [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]

    def test_user_image_becomes_input_image(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image_data": "BASE64"}],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert out == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,BASE64",
                    }
                ],
            }
        ]

    def test_user_plain_string_content_passes_through(self):
        msgs = [{"role": "user", "content": "plain"}]
        out = translate_messages_to_openai_responses(msgs)
        assert out == [{"role": "user", "content": "plain"}]

    def test_assistant_tool_use_becomes_top_level_function_call(self):
        """Responses' ``input`` is flat: an assistant tool call is a
        top-level ``function_call`` item, not nested inside an
        assistant message's ``tool_calls`` field."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "python_action",
                        "input": {"title": "t", "code": "x"},
                    }
                ],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert out == [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "python_action",
                "arguments": json.dumps({"title": "t", "code": "x"}),
            }
        ]

    def test_assistant_text_becomes_assistant_message_with_output_text(self):
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "all done"}],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert out == [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "all done"}],
            }
        ]

    def test_assistant_thinking_with_responses_signature_becomes_reasoning_item(self):
        sig = encode_reasoning_signature("rs_abc", "enc-payload")
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "signature": sig}],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert out == [
            {
                "type": "reasoning",
                "id": "rs_abc",
                "encrypted_content": "enc-payload",
                "summary": [],
            }
        ]

    def test_thinking_block_with_non_responses_signature_dropped(self):
        """A thinking block whose signature came from Gemini (raw bytes
        without our tag) is dropped — Responses can't use it and we
        shouldn't feed the API an id-less reasoning item."""
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "signature": b"\x01gemini"}],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert out == []

    def test_assistant_mixed_order_preserved(self):
        """An assistant turn with reasoning + tool_use + text fans out
        into three top-level items in the original order."""
        sig = encode_reasoning_signature("rs_1", "enc")
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "signature": sig},
                    {
                        "type": "tool_use",
                        "id": "call_x",
                        "name": "write_file",
                        "input": {"path": "/x", "content": "y"},
                    },
                    {"type": "text", "text": "wrote it"},
                ],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert [item.get("type") or item.get("role") for item in out] == [
            "reasoning",
            "function_call",
            "assistant",
        ]

    def test_tool_result_becomes_function_call_output(self):
        """Responses expects tool results as top-level
        ``function_call_output`` items, not as a ``role: "tool"``
        message."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "tool output here",
                    }
                ],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert out == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "tool output here",
            }
        ]

    def test_tool_result_with_image_flattens_placeholder(self):
        """Responses function_call_output.output is a single string;
        image parts flatten to ``[image]`` placeholders."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {"type": "image", "image_data": "B64"},
                        ],
                    }
                ],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert out[0]["output"] == "ok\n[image]"

    def test_user_message_with_mixed_tool_results_and_text(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "r1",
                    },
                    {"type": "text", "text": "follow-up prompt"},
                ],
            }
        ]
        out = translate_messages_to_openai_responses(msgs)
        assert [item.get("type") or item.get("role") for item in out] == [
            "function_call_output",
            "user",
        ]
        assert out[1]["content"] == [{"type": "input_text", "text": "follow-up prompt"}]


# ---------------------------------------------------------------------------
# Streaming translation
# ---------------------------------------------------------------------------


class TestResponsesStreamTranslation:
    def test_function_call_emits_start_delta_end(self):
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "python_action",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": '{"code":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": '"pass"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "python_action",
                    "arguments": '{"code":"pass"}',
                },
            },
            {"type": "response.completed", "response": {"usage": {}}},
        ]
        out = list(translate_openai_responses_stream_to_events(iter(events)))
        starts = [e for e in out if isinstance(e, ToolCallStart)]
        deltas = [e for e in out if isinstance(e, ToolCallArgDelta)]
        ends = [e for e in out if isinstance(e, ToolCallEnd)]
        assert len(starts) == 1
        assert starts[0].call_id == "call_1"
        assert starts[0].tool_name == "python_action"
        # Args concatenate via the parser downstream; adapter just
        # forwards the raw delta strings.
        assert "".join(d.argument_chunk for d in deltas) == '{"code":"pass"}'
        assert len(ends) == 1 and ends[0].call_id == "call_1"

    def test_reasoning_item_becomes_thinking_part_with_signature(self):
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1"},
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "delta": "step one.",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "delta": " step two.",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [
                        {"type": "summary_text", "text": "step one. step two."}
                    ],
                    "encrypted_content": "ENC",
                },
            },
        ]
        out = list(translate_openai_responses_stream_to_events(iter(events)))
        thinking = [e for e in out if isinstance(e, ThinkingPart)]
        assert len(thinking) == 1
        assert thinking[0].text == "step one. step two."
        decoded = decode_reasoning_signature(thinking[0].signature)
        assert decoded == ("rs_1", "ENC")

    def test_reasoning_with_encrypted_but_no_text_is_redacted(self):
        """When encrypted_content is present but no summary text
        streams (redacted reasoning), emit a redacted ThinkingPart so
        it still round-trips."""
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_2"},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_2",
                    "summary": [],
                    "encrypted_content": "ENC2",
                },
            },
        ]
        out = list(translate_openai_responses_stream_to_events(iter(events)))
        thinking = [e for e in out if isinstance(e, ThinkingPart)]
        assert len(thinking) == 1
        assert thinking[0].redacted is True
        assert thinking[0].text is None
        decoded = decode_reasoning_signature(thinking[0].signature)
        assert decoded == ("rs_2", "ENC2")

    def test_message_item_becomes_text_part(self):
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1", "role": "assistant"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello ",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "world",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello world"}],
                },
            },
        ]
        out = list(translate_openai_responses_stream_to_events(iter(events)))
        texts = [e for e in out if isinstance(e, TextPart)]
        assert len(texts) == 1
        assert texts[0].text == "hello world"

    def test_usage_captured_on_completed(self):
        events = [
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 42,
                        "output_tokens": 7,
                        "input_tokens_details": {"cached_tokens": 20},
                    }
                },
            }
        ]
        usage: dict = {}
        list(
            translate_openai_responses_stream_to_events(
                iter(events), usage_holder=usage
            )
        )
        assert usage["input_tokens"] == 42
        assert usage["output_tokens"] == 7
        assert usage["cached_tokens"] == 20

    def test_multiple_output_items_preserve_order(self):
        """A full turn: reasoning → function_call → message all flow
        through in the order the server emitted them."""
        events = [
            {
                "type": "response.output_item.added",
                "item": {"type": "reasoning", "id": "rs_1"},
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "think"}],
                    "encrypted_content": "E",
                },
            },
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "python_action",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": "{}",
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "python_action",
                    "arguments": "{}",
                },
            },
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_1", "role": "assistant"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "done",
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            },
        ]
        out = list(translate_openai_responses_stream_to_events(iter(events)))
        kinds = []
        for e in out:
            if isinstance(e, ThinkingPart):
                kinds.append("thinking")
            elif isinstance(e, ToolCallStart):
                kinds.append("tool_start")
            elif isinstance(e, ToolCallEnd):
                kinds.append("tool_end")
            elif isinstance(e, TextPart):
                kinds.append("text")
        assert kinds == ["thinking", "tool_start", "tool_end", "text"]


@pytest.mark.asyncio
async def test_async_stream_translator():
    """Same input, async iterator, mirrors sync behaviour."""

    async def gen():
        for e in [
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_a",
                    "call_id": "call_a",
                    "name": "python_action",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_a",
                "delta": "{}",
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_a",
                    "call_id": "call_a",
                    "arguments": "{}",
                },
            },
        ]:
            yield e

    out: list = []
    async for ev in atranslate_openai_responses_stream_to_events(gen()):
        out.append(ev)
    assert [type(e).__name__ for e in out] == [
        "ToolCallStart",
        "ToolCallArgDelta",
        "ToolCallEnd",
    ]
