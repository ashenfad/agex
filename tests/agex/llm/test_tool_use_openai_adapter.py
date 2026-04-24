"""Tests for the OpenAI adapter in agex.llm.formats.tool_use."""

import json

import pytest

from agex.llm.formats.tool_use import (
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallStart,
)
from agex.llm.formats.tool_use.openai_adapter import (
    atranslate_openai_stream_to_events,
    decode_openrouter_reasoning,
    encode_openrouter_reasoning,
    schemas_to_openai_tools,
    translate_messages_to_openai,
    translate_openai_stream_to_events,
)


class TestSchemasToOpenAITools:
    def test_wraps_each_schema_in_function_envelope(self):
        schemas = [
            {
                "name": "python_action",
                "description": "Run Python.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "write_file",
                "description": "Write a file.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
        tools = schemas_to_openai_tools(schemas)
        assert [t["type"] for t in tools] == ["function", "function"]
        assert tools[0]["function"]["name"] == "python_action"
        assert tools[0]["function"]["description"] == "Run Python."
        assert tools[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


class TestTranslateMessagesToOpenAI:
    def test_plain_user_text_message_passes_through(self):
        msgs = [{"role": "user", "content": "hello"}]
        out = translate_messages_to_openai(msgs)
        assert out == [{"role": "user", "content": "hello"}]

    def test_user_text_parts_flatten_to_string(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "[1] task one"}],
            }
        ]
        out = translate_messages_to_openai(msgs)
        assert out == [{"role": "user", "content": "[1] task one"}]

    def test_assistant_text_plus_tool_use_both_replayed(self):
        """A turn that mixed a TextEmission with a PythonEmission must
        come back out as an assistant message carrying both the
        ``content`` string AND the ``tool_calls`` array — otherwise
        the model's own prose disappears on replay."""
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
        out = translate_messages_to_openai(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "working on it"
        assert len(out[0]["tool_calls"]) == 1
        assert out[0]["tool_calls"][0]["function"]["name"] == "python_action"

    def test_assistant_tool_use_becomes_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "python_action",
                        "input": {
                            "title": "t",
                            "thinking": "T",
                            "code": "print(1)",
                        },
                    }
                ],
            }
        ]
        out = translate_messages_to_openai(msgs)
        assert len(out) == 1
        asst = out[0]
        assert asst["role"] == "assistant"
        assert asst["content"] is None
        assert len(asst["tool_calls"]) == 1
        tc = asst["tool_calls"][0]
        assert tc["id"] == "toolu_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "python_action"
        assert json.loads(tc["function"]["arguments"]) == {
            "title": "t",
            "thinking": "T",
            "code": "print(1)",
        }

    def test_assistant_multiple_tool_uses_preserve_order(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_a",
                        "name": "write_file",
                        "input": {"path": "/x.py", "content": "X"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_b",
                        "name": "python_action",
                        "input": {
                            "title": "t",
                            "thinking": "T",
                            "code": "import x",
                        },
                    },
                ],
            }
        ]
        out = translate_messages_to_openai(msgs)
        names = [c["function"]["name"] for c in out[0]["tool_calls"]]
        assert names == ["write_file", "python_action"]

    def test_user_tool_results_become_separate_tool_messages(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_a",
                        "content": "ok",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_b",
                        "content": "hello world",
                    },
                ],
            }
        ]
        out = translate_messages_to_openai(msgs)
        assert [m["role"] for m in out] == ["tool", "tool"]
        assert out[0]["tool_call_id"] == "toolu_a"
        assert out[0]["content"] == "ok"
        assert out[1]["tool_call_id"] == "toolu_b"
        assert out[1]["content"] == "hello world"

    def test_mixed_tool_results_and_text_in_user_message(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_main",
                        "content": "5",
                    },
                    {"type": "text", "text": "[2] next task"},
                ],
            }
        ]
        out = translate_messages_to_openai(msgs)
        assert [m["role"] for m in out] == ["tool", "user"]
        assert out[0]["tool_call_id"] == "toolu_main"
        assert out[1]["content"] == "[2] next task"

    def test_tool_result_with_image_parts_becomes_placeholder_text(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_main",
                        "content": [
                            {"type": "text", "text": "plot below"},
                            {"type": "image", "image_data": "b64..."},
                        ],
                    },
                ],
            }
        ]
        out = translate_messages_to_openai(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "tool"
        assert "plot below" in out[0]["content"]
        assert "[image]" in out[0]["content"]

    def test_system_message_passes_through(self):
        msgs = [{"role": "system", "content": "be helpful"}]
        assert translate_messages_to_openai(msgs) == msgs


class TestTranslateOpenAIStream:
    def test_single_tool_call(self):
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": '{"ti',
                                    },
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'tle":"t"}'},
                                }
                            ]
                        },
                    }
                ],
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = list(translate_openai_stream_to_events(iter(chunks)))
        assert events == [
            ToolCallStart(call_id="call_1", tool_name="python_action"),
            ToolCallArgDelta(call_id="call_1", argument_chunk='{"ti'),
            ToolCallArgDelta(call_id="call_1", argument_chunk='tle":"t"}'),
            ToolCallEnd(call_id="call_1"),
        ]

    def test_multiple_parallel_tool_calls(self):
        # Two tool calls, interleaved deltas across indices 0 and 1.
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": "",
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": "",
                                    },
                                },
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"p":"x"}'}},
                                {"index": 1, "function": {"arguments": '{"c":"y"}'}},
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = list(translate_openai_stream_to_events(iter(chunks)))
        # Both starts, both arg deltas, both ends.
        starts = [e for e in events if isinstance(e, ToolCallStart)]
        ends = [e for e in events if isinstance(e, ToolCallEnd)]
        deltas = [e for e in events if isinstance(e, ToolCallArgDelta)]
        assert {s.tool_name for s in starts} == {"write_file", "python_action"}
        assert {e.call_id for e in ends} == {"call_a", "call_b"}
        assert len(deltas) == 2

    def test_captures_usage_from_final_chunk(self):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 7}},
        ]
        usage_holder: dict = {}
        list(translate_openai_stream_to_events(iter(chunks), usage_holder=usage_holder))
        assert usage_holder["input_tokens"] == 42
        assert usage_holder["output_tokens"] == 7

    def test_empty_args_do_not_emit_delta(self):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "python_action"},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = list(translate_openai_stream_to_events(iter(chunks)))
        # Start + End, no ArgDeltas.
        assert [type(e).__name__ for e in events] == [
            "ToolCallStart",
            "ToolCallEnd",
        ]

    def test_plain_text_content_becomes_textpart(self):
        """Plain ``delta.content`` chunks (pure text response, no tool
        calls) must surface as a TextPart so the turn isn't a silent
        zero-emission black hole."""
        from agex.llm.formats.tool_use.events import TextPart

        chunks = [
            {"choices": [{"delta": {"content": "I'll think "}}]},
            {"choices": [{"delta": {"content": "about it."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        events = list(translate_openai_stream_to_events(iter(chunks)))
        texts = [e for e in events if isinstance(e, TextPart)]
        assert len(texts) == 1
        assert texts[0].text == "I'll think about it."

    def test_text_alongside_tool_call(self):
        """Text chunks + a tool call in the same turn: both surface."""
        from agex.llm.formats.tool_use.events import TextPart

        chunks = [
            {"choices": [{"delta": {"content": "calling now: "}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = list(translate_openai_stream_to_events(iter(chunks)))
        assert any(
            isinstance(e, TextPart) and e.text == "calling now: " for e in events
        )
        # Text flushes after the tool call (buffered to stream end).
        assert [type(e).__name__ for e in events] == [
            "ToolCallStart",
            "ToolCallArgDelta",
            "ToolCallEnd",
            "TextPart",
        ]

    def test_sdk_model_dump_normalization(self):
        """Chunks with ``model_dump`` (pydantic models) should work too."""

        class FakeChunk:
            def __init__(self, d):
                self._d = d

            def model_dump(self):
                return self._d

        chunks = [
            FakeChunk(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "function": {
                                            "name": "python_action",
                                            "arguments": '{"x":1}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
        ]
        events = list(translate_openai_stream_to_events(iter(chunks)))
        assert [type(e).__name__ for e in events] == [
            "ToolCallStart",
            "ToolCallArgDelta",
            "ToolCallEnd",
        ]


class TestOpenRouterReasoningRoundTrip:
    """End-to-end: stream reasoning_details off the wire, build a
    :class:`ThinkingEmission`, render it back as an assistant
    message and verify ``reasoning_details`` matches byte-for-byte.
    """

    def test_stream_capture_into_message_reasoning_details(self):
        from agex.agent.emissions import ThinkingEmission
        from agex.llm.core import EmissionsBuilder

        source_details = [
            {
                "type": "reasoning.summary",
                "format": "anthropic-claude-v1",
                "id": "block_1",
                "index": 0,
                "text": "OK, let me think about this. ",
            },
            {
                "type": "reasoning.summary",
                "format": "anthropic-claude-v1",
                "id": "block_1",
                "index": 0,
                "text": "The user wants a prime-finding fn.",
            },
        ]
        chunks = [
            {"choices": [{"delta": {"reasoning_details": [source_details[0]]}}]},
            {"choices": [{"delta": {"reasoning_details": [source_details[1]]}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]

        # Feed the translator + builder end-to-end.
        builder = EmissionsBuilder(agent_name="a")
        from agex.llm.formats.tool_use import parse_tool_events

        tool_events = translate_openai_stream_to_events(iter(chunks))
        for token in parse_tool_events(tool_events):
            builder.process_token(token)
        resp = builder.build()

        # One ThinkingEmission with the aggregated text.
        thinking_ems = [em for em in resp.emissions if isinstance(em, ThinkingEmission)]
        assert len(thinking_ems) == 1
        assert (
            thinking_ems[0].text
            == "OK, let me think about this. The user wants a prime-finding fn."
        )

        # Render the captured ThinkingEmission back through the
        # tool-use renderer + translate_messages_to_openai and
        # verify ``reasoning_details`` survives the round-trip.
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "text": thinking_ems[0].text,
                        "signature": thinking_ems[0].signature,
                    }
                ],
            }
        ]
        out = translate_messages_to_openai(msgs)
        # One reasoning block (the two deltas coalesced at index 0).
        assert len(out[0]["reasoning_details"]) == 1
        # Concatenated text survives byte-for-byte.
        assert out[0]["reasoning_details"][0]["text"] == thinking_ems[0].text
        # Non-text fields (format, id, type, index) preserved.
        assert out[0]["reasoning_details"][0]["format"] == "anthropic-claude-v1"
        assert out[0]["reasoning_details"][0]["id"] == "block_1"
        assert out[0]["reasoning_details"][0]["type"] == "reasoning.summary"
        assert out[0]["reasoning_details"][0]["index"] == 0


class TestOpenRouterReasoningCodec:
    """The ``reasoning_details`` array rides on
    :class:`ThinkingEmission.signature` bytes with an
    ``openrouter-reasoning:`` tag prefix.  Every byte of the original
    array must round-trip — OpenRouter requires the sequence of
    reasoning blocks replayed on subsequent turns to match exactly."""

    def test_round_trip_preserves_array(self):
        details = [
            {
                "type": "reasoning.summary",
                "format": "anthropic-claude-v1",
                "id": "block_1",
                "index": 0,
                "text": "let me think...",
            },
            {
                "type": "reasoning.encrypted",
                "format": "anthropic-claude-v1",
                "id": "block_1",
                "index": 1,
                "data": "ENCRYPTED_BLOB",
            },
        ]
        sig = encode_openrouter_reasoning(details)
        assert decode_openrouter_reasoning(sig) == details

    def test_other_provider_signatures_decode_none(self):
        """Gemini thought_signature bytes / Responses openai-encoded
        signatures / Anthropic raw strings must fail-closed when
        handed to this decoder so they aren't forwarded to
        OpenRouter as garbage reasoning_details."""
        assert decode_openrouter_reasoning(b"\x01\x02gemini") is None
        assert decode_openrouter_reasoning(b'openai-responses:{"id":"rs"}') is None
        assert decode_openrouter_reasoning("not-bytes") is None  # type: ignore[arg-type]


class TestReasoningDetailsOnAssistantMessage:
    """When an assistant ThinkingEmission carries an OpenRouter-encoded
    signature, ``translate_messages_to_openai`` must unpack the array
    onto ``message.reasoning_details`` so the next turn can replay
    it.  Signatures from other providers are silently ignored on
    this path — those are for their own adapters."""

    def test_reasoning_details_attached_to_assistant_message(self):
        details = [
            {
                "type": "reasoning.summary",
                "format": "anthropic-claude-v1",
                "id": "block_1",
                "index": 0,
                "text": "thinking...",
            }
        ]
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "signature": encode_openrouter_reasoning(details),
                    },
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "python_action",
                        "input": {"code": "pass"},
                    },
                ],
            }
        ]
        out = translate_messages_to_openai(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert out[0]["reasoning_details"] == details
        # Tool calls still land in the ``tool_calls`` field, not
        # interleaved with reasoning.
        assert out[0]["tool_calls"][0]["function"]["name"] == "python_action"

    def test_non_openrouter_signature_dropped(self):
        """A thinking block whose signature came from Gemini
        (``thought_signature`` bytes, no tag prefix) doesn't produce
        a ``reasoning_details`` field on the OpenAI/OpenRouter
        message."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "signature": b"\x01\x02gemini"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "python_action",
                        "input": {},
                    },
                ],
            }
        ]
        out = translate_messages_to_openai(msgs)
        assert "reasoning_details" not in out[0]


class TestReasoningDetailsStreamCapture:
    """Streaming deltas with ``reasoning_details`` accumulate by
    ``index`` and flush as a single :class:`ThinkingPart` at stream
    end, carrying the full array in the signature."""

    def test_accumulates_text_across_deltas(self):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {
                                    "type": "reasoning.summary",
                                    "format": "anthropic-claude-v1",
                                    "id": "block_1",
                                    "index": 0,
                                    "text": "first half. ",
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {
                                    "type": "reasoning.summary",
                                    "format": "anthropic-claude-v1",
                                    "id": "block_1",
                                    "index": 0,
                                    "text": "second half.",
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        out = list(translate_openai_stream_to_events(iter(chunks)))
        thinking = [e for e in out if isinstance(e, ThinkingPart)]
        assert len(thinking) == 1
        # Visible text aggregates.
        assert thinking[0].text == "first half. second half."
        # Signature round-trips the full array with concatenated text.
        details = decode_openrouter_reasoning(thinking[0].signature)
        assert details is not None
        assert len(details) == 1
        assert details[0]["text"] == "first half. second half."
        assert details[0]["format"] == "anthropic-claude-v1"
        assert thinking[0].redacted is False

    def test_multiple_indices_sorted_and_preserved(self):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {
                                    "type": "reasoning.summary",
                                    "format": "anthropic-claude-v1",
                                    "id": "b0",
                                    "index": 0,
                                    "text": "step one",
                                },
                                {
                                    "type": "reasoning.encrypted",
                                    "format": "anthropic-claude-v1",
                                    "id": "b1",
                                    "index": 1,
                                    "data": "ENC",
                                },
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        out = list(translate_openai_stream_to_events(iter(chunks)))
        thinking = [e for e in out if isinstance(e, ThinkingPart)]
        assert len(thinking) == 1
        details = decode_openrouter_reasoning(thinking[0].signature)
        assert [d["index"] for d in details] == [0, 1]
        assert details[1]["type"] == "reasoning.encrypted"
        assert details[1]["data"] == "ENC"

    def test_encrypted_only_block_renders_redacted(self):
        """A reasoning block with encrypted content but no visible
        text should become a redacted ThinkingPart — the opaque
        bytes still round-trip via the signature."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {
                                    "type": "reasoning.encrypted",
                                    "format": "openai-responses-v1",
                                    "id": "r1",
                                    "index": 0,
                                    "data": "OPAQUE",
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        out = list(translate_openai_stream_to_events(iter(chunks)))
        thinking = [e for e in out if isinstance(e, ThinkingPart)]
        assert len(thinking) == 1
        assert thinking[0].redacted is True
        assert thinking[0].text is None
        details = decode_openrouter_reasoning(thinking[0].signature)
        assert details[0]["data"] == "OPAQUE"


@pytest.mark.asyncio
async def test_async_stream_translator():
    async def gen():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {
                                    "name": "python_action",
                                    "arguments": '{"x":',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    out = []
    async for ev in atranslate_openai_stream_to_events(gen()):
        out.append(ev)
    assert [type(e).__name__ for e in out] == [
        "ToolCallStart",
        "ToolCallArgDelta",
        "ToolCallArgDelta",
        "ToolCallEnd",
    ]
