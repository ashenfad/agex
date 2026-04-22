"""Tests for the OpenAI adapter in agex.llm.formats.tool_use."""

import json

import pytest

from agex.llm.formats.tool_use import ToolCallArgDelta, ToolCallEnd, ToolCallStart
from agex.llm.formats.tool_use.openai_adapter import (
    atranslate_openai_stream_to_events,
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
