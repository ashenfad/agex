"""Tests for the Anthropic adapter in agex.llm.formats.tool_use."""

import pytest

from agex.llm.formats.tool_use import (
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallStart,
)
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

    def test_assistant_tool_use_strips_signature(self):
        """The renderer attaches ``signature`` to tool_use blocks for
        Gemini's ``thought_signature`` round-trip.  Anthropic's Messages
        API doesn't accept that field on tool_use blocks (it lives on
        separate ``thinking`` blocks instead) and would 400.  Strip on
        egress so cross-provider replay stays safe."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "python_action",
                        "input": {"code": "pass"},
                        "signature": b"opaque-bytes-from-gemini",
                    }
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        assert out[0]["content"][0] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "python_action",
            "input": {"code": "pass"},
        }
        # Original message must not be mutated.
        assert "signature" in msgs[0]["content"][0]

    def test_assistant_text_plus_tool_use_both_replayed(self):
        """A turn that mixed a TextEmission with a PythonEmission must
        pass through as assistant content containing both the text
        block and the tool_use block, in order — Anthropic natively
        accepts interleaved text + tool_use."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "working on it"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "python_action",
                        "input": {"code": "pass"},
                    },
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        assert out[0]["role"] == "assistant"
        assert out[0]["content"][0] == {"type": "text", "text": "working on it"}
        assert out[0]["content"][1]["type"] == "tool_use"
        assert out[0]["content"][1]["name"] == "python_action"

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

    def test_text_content_block_captured_as_textpart(self):
        """Claude can emit a text block alongside tool_use blocks (and
        providers in general may respond with plain text).  Capture
        it as a TextPart so the turn isn't blank in the event log and
        the model sees its own words on the next request."""
        from agex.llm.formats.tool_use.events import TextPart

        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello "},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "world"},
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
        texts = [e for e in out if isinstance(e, TextPart)]
        assert len(texts) == 1
        assert texts[0].text == "hello world"
        tool_events = [type(e).__name__ for e in out if not isinstance(e, TextPart)]
        assert tool_events == ["ToolCallStart", "ToolCallArgDelta", "ToolCallEnd"]

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


class TestThinkingBlocks:
    """Claude's extended-thinking blocks arrive as their own
    content-block type, with text streamed via ``thinking_delta`` and
    signature bytes via ``signature_delta`` — both keyed by the same
    content-block ``index``.  We must capture them, emit one
    :class:`ThinkingPart` per block when the block closes, and
    round-trip them on replay so Claude can continue coherently."""

    def test_single_thinking_block_emits_thinking_part(self):
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Step 1. "},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Step 2."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "deadbeef"},
            },
            {"type": "content_block_stop", "index": 0},
        ]
        out = list(translate_anthropic_stream_to_events(iter(events)))
        thinking_parts = [e for e in out if isinstance(e, ThinkingPart)]
        assert len(thinking_parts) == 1
        tp = thinking_parts[0]
        assert tp.text == "Step 1. Step 2."
        assert tp.signature == b"deadbeef"
        assert tp.redacted is False

    def test_redacted_thinking_preserves_data_payload(self):
        """Redacted blocks ship the opaque signed payload as ``data``
        on the initial block_start; no deltas.  We stash it in the
        ``signature`` bytes so the renderer can round-trip it as
        ``{"type": "redacted_thinking", "data": ...}``."""
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "redacted_thinking",
                    "data": "opaque-encrypted-blob",
                },
            },
            {"type": "content_block_stop", "index": 0},
        ]
        out = list(translate_anthropic_stream_to_events(iter(events)))
        thinking_parts = [e for e in out if isinstance(e, ThinkingPart)]
        assert len(thinking_parts) == 1
        tp = thinking_parts[0]
        assert tp.redacted is True
        assert tp.text is None
        assert tp.signature == b"opaque-encrypted-blob"

    def test_thinking_interleaved_with_tool_uses(self):
        """Claude 4.x extended thinking can interleave thinking blocks
        with tool_use blocks.  Preserve stream order so the parser
        assigns emission indices in the order the model emitted them."""
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "plan A"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "SIG_A"},
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
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "thinking"},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "thinking_delta", "thinking": "plan B"},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "signature_delta", "signature": "SIG_B"},
            },
            {"type": "content_block_stop", "index": 2},
        ]
        out = list(translate_anthropic_stream_to_events(iter(events)))
        # Preserve order: thinking → tool_use start/delta/end → thinking.
        kinds = [type(e).__name__ for e in out]
        assert kinds == [
            "ThinkingPart",
            "ToolCallStart",
            "ToolCallArgDelta",
            "ToolCallEnd",
            "ThinkingPart",
        ]
        assert out[0].text == "plan A"
        assert out[0].signature == b"SIG_A"
        assert out[4].text == "plan B"
        assert out[4].signature == b"SIG_B"

    def test_dangling_thinking_gets_safety_emit(self):
        """If the stream ends without a content_block_stop (shouldn't
        happen, but defensive), we still emit the accumulated thinking
        so the signature isn't silently lost."""
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "hi"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "SIG"},
            },
            # No content_block_stop.
        ]
        out = list(translate_anthropic_stream_to_events(iter(events)))
        thinking_parts = [e for e in out if isinstance(e, ThinkingPart)]
        assert len(thinking_parts) == 1
        assert thinking_parts[0].text == "hi"


class TestThinkingBlockReplay:
    """``translate_messages_to_anthropic`` must turn our generic
    ``thinking`` blocks (from signed ThinkingEmissions) back into
    Anthropic's native ``thinking`` / ``redacted_thinking`` content
    blocks — the Messages API requires them to be replayed verbatim
    with their original signature for Claude to continue."""

    def test_thinking_block_becomes_native_thinking(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "text": "my reasoning",
                        "signature": b"deadbeef",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "python_action",
                        "input": {},
                    },
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        blocks = out[0]["content"]
        assert blocks[0] == {
            "type": "thinking",
            "thinking": "my reasoning",
            "signature": "deadbeef",
        }
        # Tool use follows untouched.
        assert blocks[1]["type"] == "tool_use"

    def test_redacted_thinking_block_becomes_native_redacted(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "signature": b"opaque-blob",
                        "redacted": True,
                    },
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        assert out[0]["content"] == [
            {"type": "redacted_thinking", "data": "opaque-blob"},
        ]

    def test_thinking_block_with_string_signature_passes_through(self):
        """Sanity: if a caller happens to hand a string signature
        (shouldn't happen from our pipeline — emissions always store
        bytes — but be forgiving), we don't crash."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "text": "hi",
                        "signature": "already-a-string",
                    },
                ],
            }
        ]
        out = translate_messages_to_anthropic(msgs)
        assert out[0]["content"][0]["signature"] == "already-a-string"


class TestThinkingFullRoundTrip:
    """Stream → parser → builder → renderer → translate_messages_to_anthropic.
    Signed Claude thinking blocks should survive every link and come
    out the other side as ``{"type": "thinking", "thinking": ...,
    "signature": ...}`` — the exact shape Claude requires on replay."""

    def test_thinking_plus_tool_use_round_trip(self):
        from agex.agent.emissions import PythonEmission, ThinkingEmission
        from agex.agent.events import ActionEvent
        from agex.llm.core import EmissionsBuilder
        from agex.llm.formats.tool_use.parser import parse_tool_events

        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "planning..."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "S1G"},
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
                "delta": {"type": "input_json_delta", "partial_json": '{"code":"x"}'},
            },
            {"type": "content_block_stop", "index": 1},
        ]

        tool_events = translate_anthropic_stream_to_events(iter(events))
        tokens = list(parse_tool_events(tool_events))
        builder = EmissionsBuilder()
        for t in tokens:
            builder.process_token(t)
        response = builder.build()

        # Emissions landed in order: thinking, then python.
        assert len(response.emissions) == 2
        assert isinstance(response.emissions[0], ThinkingEmission)
        assert response.emissions[0].text == "planning..."
        assert response.emissions[0].signature == b"S1G"
        assert isinstance(response.emissions[1], PythonEmission)

        # Render back and re-translate: thinking block comes out as
        # Claude's native ``thinking`` shape with the signature
        # preserved verbatim.
        event = ActionEvent(agent_name="a", emissions=list(response.emissions))
        from agex.llm.formats.tool_use.renderer import render_events_as_tool_use

        rendered = render_events_as_tool_use([event])
        anthropic_msgs = translate_messages_to_anthropic(rendered)
        assistant = next(m for m in anthropic_msgs if m["role"] == "assistant")
        blocks = assistant["content"]
        assert blocks[0] == {
            "type": "thinking",
            "thinking": "planning...",
            "signature": "S1G",
        }
        assert blocks[1]["type"] == "tool_use"


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
