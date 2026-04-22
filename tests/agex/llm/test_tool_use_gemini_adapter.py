"""Tests for the Gemini adapter in agex.llm.formats.tool_use."""

import json
from types import SimpleNamespace

import pytest

from agex.llm.formats.tool_use import ToolCallArgDelta, ToolCallEnd, ToolCallStart
from agex.llm.formats.tool_use.gemini_adapter import (
    atranslate_gemini_stream_to_events,
    schemas_to_gemini_function_declarations,
    translate_gemini_stream_to_events,
    translate_messages_to_gemini,
)


class TestSchemasToGeminiFunctionDeclarations:
    def test_name_description_parameters_preserved(self):
        schemas = [
            {
                "name": "python_action",
                "description": "Run Python.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "write_file",
                "description": "Write a file.",
                "parameters": {"type": "object"},
            },
        ]
        out = schemas_to_gemini_function_declarations(schemas)
        assert [d["name"] for d in out] == ["python_action", "write_file"]
        assert out[0]["parameters"] == {"type": "object", "properties": {}}
        # No envelope (no "type": "function"), no key rename.
        assert all("type" not in d for d in out)


class TestTranslateMessagesToGemini:
    def test_user_role_preserved(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "[1] task"}]}]
        out = translate_messages_to_gemini(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        assert out[0]["parts"] == [{"text": "[1] task"}]

    def test_assistant_role_becomes_model(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "python_action",
                        "input": {"title": "t"},
                    }
                ],
            }
        ]
        out = translate_messages_to_gemini(msgs)
        assert out[0]["role"] == "model"
        assert out[0]["parts"][0]["function_call"] == {
            "id": "toolu_1",
            "name": "python_action",
            "args": {"title": "t"},
        }

    def test_tool_result_gets_name_from_preceding_tool_use(self):
        """The tool_result block only carries tool_use_id; translator
        recovers the function name from the preceding assistant
        tool_use with matching id."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_main",
                        "name": "python_action",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_main",
                        "content": "output",
                    }
                ],
            },
        ]
        out = translate_messages_to_gemini(msgs)
        assert len(out) == 2
        fr = out[1]["parts"][0]["function_response"]
        assert fr["id"] == "toolu_main"
        assert fr["name"] == "python_action"
        assert fr["response"] == {"result": "output"}

    def test_image_part_becomes_inline_data(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "photo:"},
                    {"type": "image", "image_data": "B64"},
                ],
            }
        ]
        out = translate_messages_to_gemini(msgs)
        parts = out[0]["parts"]
        assert parts[0] == {"text": "photo:"}
        assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": "B64"}}

    def test_tool_result_with_image_content_flattens_to_text(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_main",
                        "name": "python_action",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_main",
                        "content": [
                            {"type": "text", "text": "plot below"},
                            {"type": "image", "image_data": "B64"},
                        ],
                    }
                ],
            },
        ]
        out = translate_messages_to_gemini(msgs)
        fr = out[1]["parts"][0]["function_response"]
        result = fr["response"]["result"]
        assert "plot below" in result
        assert "[image]" in result

    def test_empty_text_blocks_skipped_but_part_list_preserved(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "hello"},
                ],
            }
        ]
        out = translate_messages_to_gemini(msgs)
        assert out[0]["parts"] == [{"text": "hello"}]

    def test_messages_with_no_parts_dropped(self):
        """Gemini rejects Content with empty parts — we drop entirely."""
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": [{"type": "text", "text": "real"}]},
        ]
        out = translate_messages_to_gemini(msgs)
        assert len(out) == 1
        assert out[0]["parts"] == [{"text": "real"}]

    def test_string_content_wraps_in_text_part(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = translate_messages_to_gemini(msgs)
        assert out[0]["parts"] == [{"text": "hi"}]

    def test_multiple_tool_uses_tracked_across_turns(self):
        """Two assistant tool_use blocks → two tool_result blocks
        in the following user message; each gets its correct name."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_a",
                        "name": "write_file",
                        "input": {"path": "x"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_b",
                        "name": "python_action",
                        "input": {},
                    },
                ],
            },
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
                        "content": "hi",
                    },
                ],
            },
        ]
        out = translate_messages_to_gemini(msgs)
        frs = [p["function_response"] for p in out[1]["parts"]]
        assert frs[0]["name"] == "write_file"
        assert frs[1]["name"] == "python_action"


class TestTranslateGeminiStream:
    def test_single_function_call_emits_triple(self):
        chunks = [
            SimpleNamespace(
                function_calls=[
                    SimpleNamespace(
                        id="call_1",
                        name="python_action",
                        args={"title": "t", "thinking": "T", "code": "x"},
                    )
                ],
                usage_metadata=None,
            )
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        assert len(out) == 3
        assert isinstance(out[0], ToolCallStart)
        assert out[0].call_id == "call_1"
        assert out[0].tool_name == "python_action"
        assert isinstance(out[1], ToolCallArgDelta)
        assert json.loads(out[1].argument_chunk) == {
            "title": "t",
            "thinking": "T",
            "code": "x",
        }
        assert isinstance(out[2], ToolCallEnd)
        assert out[2].call_id == "call_1"

    def test_multiple_function_calls_in_one_chunk(self):
        chunks = [
            SimpleNamespace(
                function_calls=[
                    SimpleNamespace(id="call_a", name="write_file", args={"path": "x"}),
                    SimpleNamespace(
                        id="call_b",
                        name="python_action",
                        args={"title": "t"},
                    ),
                ],
                usage_metadata=None,
            )
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        starts = [e for e in out if isinstance(e, ToolCallStart)]
        ends = [e for e in out if isinstance(e, ToolCallEnd)]
        assert [s.tool_name for s in starts] == ["write_file", "python_action"]
        assert {e.call_id for e in ends} == {"call_a", "call_b"}

    def test_duplicate_call_id_across_chunks_deduplicated(self):
        """Gemini may re-emit an already-seen function_call; translator
        should only emit Start/Delta/End for it once."""
        fc = SimpleNamespace(id="call_1", name="python_action", args={})
        chunks = [
            SimpleNamespace(function_calls=[fc], usage_metadata=None),
            SimpleNamespace(function_calls=[fc], usage_metadata=None),
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        assert len([e for e in out if isinstance(e, ToolCallStart)]) == 1
        assert len([e for e in out if isinstance(e, ToolCallEnd)]) == 1

    def test_missing_id_synthesized(self):
        chunks = [
            SimpleNamespace(
                function_calls=[
                    SimpleNamespace(id=None, name="python_action", args={}),
                    SimpleNamespace(id=None, name="write_file", args={}),
                ],
                usage_metadata=None,
            )
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        ids = [e.call_id for e in out if isinstance(e, ToolCallStart)]
        assert len(ids) == 2
        assert len(set(ids)) == 2  # unique

    def test_usage_captured(self):
        um = SimpleNamespace(prompt_token_count=42, candidates_token_count=7)
        chunks = [
            SimpleNamespace(function_calls=[], usage_metadata=um),
        ]
        usage: dict = {}
        list(translate_gemini_stream_to_events(iter(chunks), usage_holder=usage))
        assert usage["input_tokens"] == 42
        assert usage["output_tokens"] == 7

    def test_chunks_without_function_calls_produce_nothing(self):
        chunks = [
            SimpleNamespace(function_calls=None, usage_metadata=None),
            SimpleNamespace(function_calls=[], usage_metadata=None),
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        assert out == []


@pytest.mark.asyncio
async def test_async_stream_translator():
    async def gen():
        yield SimpleNamespace(
            function_calls=[
                SimpleNamespace(id="c1", name="python_action", args={"title": "t"})
            ],
            usage_metadata=None,
        )

    out = []
    async for e in atranslate_gemini_stream_to_events(gen()):
        out.append(e)
    assert [type(e).__name__ for e in out] == [
        "ToolCallStart",
        "ToolCallArgDelta",
        "ToolCallEnd",
    ]
