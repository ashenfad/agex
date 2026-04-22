"""Tests for the Anthropic adapter in agex.llm.formats.tool_use."""

import pytest

from agex.llm.formats.tool_use import ToolCallArgDelta, ToolCallEnd, ToolCallStart
from agex.llm.formats.tool_use.anthropic_adapter import (
    apply_cache_control,
    atranslate_anthropic_stream_to_events,
    schemas_to_anthropic_tools,
    translate_anthropic_stream_to_events,
    translate_messages_to_anthropic,
)


class TestSchemasToAnthropicTools:
    def test_renames_parameters_to_input_schema(self):
        schemas = [
            {
                "name": "python_action",
                "description": "Run Python.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        tools = schemas_to_anthropic_tools(schemas)
        assert tools == [
            {
                "name": "python_action",
                "description": "Run Python.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    def test_multiple_schemas(self):
        schemas = [
            {"name": "a", "description": "A", "parameters": {"type": "object"}},
            {"name": "b", "description": "B", "parameters": {"type": "object"}},
        ]
        tools = schemas_to_anthropic_tools(schemas)
        assert [t["name"] for t in tools] == ["a", "b"]
        assert all("input_schema" in t for t in tools)
        # No outer envelope like OpenAI's "type": "function".
        assert all("type" not in t for t in tools)


class TestTranslateMessagesToAnthropic:
    def test_text_only_user_message_passes_through(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "[1] do work"}],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        assert out == msgs

    def test_assistant_tool_use_passes_through(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "python_action",
                        "input": {"title": "t", "thinking": "T", "code": "x"},
                    }
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        assert out == msgs

    def test_user_tool_result_with_string_content_passes_through(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                    }
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        # tool_result is rewrapped (consistent dict shape) but content
        # preserved as string.
        assert out[0]["content"][0]["tool_use_id"] == "toolu_1"
        assert out[0]["content"][0]["content"] == "ok"

    def test_image_part_in_user_message_translates_to_source_envelope(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "plot:"},
                    {"type": "image", "image_data": "B64DATA"},
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        image_block = out[0]["content"][1]
        assert image_block["type"] == "image"
        assert image_block["source"] == {
            "type": "base64",
            "media_type": "image/png",
            "data": "B64DATA",
        }
        # The original image_data key should not leak through.
        assert "image_data" not in image_block

    def test_image_inside_tool_result_content_list_is_translated(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_main",
                        "content": [
                            {"type": "text", "text": "plot below"},
                            {"type": "image", "image_data": "IMGBYTES"},
                        ],
                    }
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        inner = out[0]["content"][0]["content"]
        assert isinstance(inner, list)
        assert inner[0] == {"type": "text", "text": "plot below"}
        assert inner[1]["type"] == "image"
        assert inner[1]["source"]["data"] == "IMGBYTES"

    def test_mixed_tool_results_and_text_preserve_order(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "out",
                    },
                    {"type": "text", "text": "[2] next"},
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        # Anthropic allows text and tool_result in the same user
        # message; order is preserved.
        blocks = out[0]["content"]
        assert [b.get("type") for b in blocks] == ["tool_result", "text"]

    def test_string_content_passes_through(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert translate_messages_to_anthropic(msgs) == msgs


class TestApplyCacheControl:
    def test_applies_to_last_block_of_target_message(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        out = apply_cache_control(msgs, cache_index=1, ttl="1h")
        # First message untouched.
        assert "cache_control" not in out[0]["content"][0]
        # Second message's last block gets cache_control.
        cc = out[1]["content"][-1].get("cache_control")
        assert cc == {"type": "ephemeral", "ttl": "1h"}

    def test_out_of_range_index_ignored(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        out = apply_cache_control(msgs, cache_index=99)
        assert out == msgs

    def test_string_content_promoted_to_text_block(self):
        msgs = [{"role": "user", "content": "hello"}]
        out = apply_cache_control(msgs, cache_index=0)
        assert out[0]["content"] == [
            {
                "type": "text",
                "text": "hello",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]


class TestTranslateAnthropicStream:
    def test_single_tool_call(self):
        events = [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 10, "output_tokens": 0}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "python_action",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"ti'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'tle":"t"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
        usage: dict = {}
        out = list(
            translate_anthropic_stream_to_events(iter(events), usage_holder=usage)
        )
        assert out == [
            ToolCallStart(call_id="toolu_1", tool_name="python_action"),
            ToolCallArgDelta(call_id="toolu_1", argument_chunk='{"ti'),
            ToolCallArgDelta(call_id="toolu_1", argument_chunk='tle":"t"}'),
            ToolCallEnd(call_id="toolu_1"),
        ]
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5

    def test_multiple_tool_calls_distinct_indices(self):
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_a",
                    "name": "write_file",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"p":"x"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_b",
                    "name": "python_action",
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"c":"y"}'},
            },
            {"type": "content_block_stop", "index": 1},
        ]
        out = list(translate_anthropic_stream_to_events(iter(events)))
        starts = [e for e in out if isinstance(e, ToolCallStart)]
        ends = [e for e in out if isinstance(e, ToolCallEnd)]
        assert [s.call_id for s in starts] == ["toolu_a", "toolu_b"]
        assert [e.call_id for e in ends] == ["toolu_a", "toolu_b"]

    def test_text_content_block_ignored(self):
        """Model sometimes emits a text block before/alongside tool_use —
        we ignore text entirely in this adapter (callers that need it
        should use the text path)."""
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "python_action",
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
            {"type": "content_block_stop", "index": 1},
        ]
        out = list(translate_anthropic_stream_to_events(iter(events)))
        # Only tool_use events emitted; text block is silent.
        assert [type(e).__name__ for e in out] == [
            "ToolCallStart",
            "ToolCallArgDelta",
            "ToolCallEnd",
        ]

    def test_usage_sums_cache_buckets(self):
        events = [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 5,
                        "cache_read_input_tokens": 100,
                        "output_tokens": 0,
                    }
                },
            },
        ]
        usage: dict = {}
        list(translate_anthropic_stream_to_events(iter(events), usage_holder=usage))
        assert usage["input_tokens"] == 115

    def test_unclosed_tool_use_block_gets_safety_end(self):
        """Stream ends without content_block_stop — translator emits
        a synthetic ToolCallEnd so downstream consumers see a clean
        close."""
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_orphan",
                    "name": "python_action",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
            # No content_block_stop, stream ends here.
        ]
        out = list(translate_anthropic_stream_to_events(iter(events)))
        assert any(
            isinstance(e, ToolCallEnd) and e.call_id == "toolu_orphan" for e in out
        )

    def test_sdk_model_dump_normalization(self):
        class FakeEvent:
            def __init__(self, d):
                self._d = d

            def model_dump(self):
                return self._d

        events = [
            FakeEvent(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "python_action",
                    },
                }
            ),
            FakeEvent(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"x":1}',
                    },
                }
            ),
            FakeEvent({"type": "content_block_stop", "index": 0}),
        ]
        out = list(translate_anthropic_stream_to_events(iter(events)))
        assert [type(e).__name__ for e in out] == [
            "ToolCallStart",
            "ToolCallArgDelta",
            "ToolCallEnd",
        ]


@pytest.mark.asyncio
async def test_async_stream_translator():
    async def gen():
        yield {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "python_action",
            },
        }
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        }
        yield {"type": "content_block_stop", "index": 0}

    out = []
    async for e in atranslate_anthropic_stream_to_events(gen()):
        out.append(e)
    assert [type(e).__name__ for e in out] == [
        "ToolCallStart",
        "ToolCallArgDelta",
        "ToolCallEnd",
    ]
