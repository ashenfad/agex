"""Tests for the Gemini adapter in agex.llm.formats.tool_use."""

import json
from types import SimpleNamespace
from typing import Any

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

    def test_tool_use_signature_becomes_part_thought_signature(self):
        """When a rendered tool_use block carries a ``signature`` (the
        bytes captured from a previous Gemini turn), the translator
        must put it onto the wrapping ``Part`` as
        ``thought_signature`` — a sibling of ``function_call``, not
        inside it.  Gemini's SDK pydantic model rejects the field on
        FunctionCall; it belongs on the Part.  Gemini 3 400s on
        function_calls missing the signature on subsequent turns."""
        sig = b"\x01\x02opaque-signature"
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "write_file",
                        "input": {"path": "x", "content": "y"},
                        "signature": sig,
                    }
                ],
            }
        ]
        out = translate_messages_to_gemini(msgs)
        part = out[0]["parts"][0]
        assert part["thought_signature"] == sig
        # Signature must NOT leak into the function_call body.
        fc = part["function_call"]
        assert "thought_signature" not in fc
        # Round-trip sanity: id/name/args still present.
        assert fc["id"] == "toolu_1"
        assert fc["name"] == "write_file"
        assert fc["args"] == {"path": "x", "content": "y"}

    def test_tool_use_without_signature_omits_field(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "python_action",
                        "input": {},
                    }
                ],
            }
        ]
        out = translate_messages_to_gemini(msgs)
        part = out[0]["parts"][0]
        assert "thought_signature" not in part
        assert "thought_signature" not in part["function_call"]

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


def _chunk(*parts: Any, usage_metadata: Any = None) -> SimpleNamespace:
    """Build a mock Gemini stream chunk whose candidate content holds
    the given parts.  Each ``part`` is already a ``SimpleNamespace``
    mimicking ``Part(function_call=..., thought_signature=...)``.
    """
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=list(parts))),
        ],
        usage_metadata=usage_metadata,
    )


def _fc_part(
    *,
    id: str | None = None,
    name: str = "python_action",
    args: dict | None = None,
    thought_signature: bytes | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        function_call=SimpleNamespace(id=id, name=name, args=args or {}),
        thought_signature=thought_signature,
    )


class TestTranslateGeminiStream:
    def test_single_function_call_emits_triple(self):
        chunks = [
            _chunk(
                _fc_part(
                    id="call_1",
                    name="python_action",
                    args={"title": "t", "thinking": "T", "code": "x"},
                )
            )
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        assert len(out) == 3
        assert isinstance(out[0], ToolCallStart)
        assert out[0].call_id == "call_1"
        assert out[0].tool_name == "python_action"
        assert out[0].signature is None
        assert isinstance(out[1], ToolCallArgDelta)
        assert json.loads(out[1].argument_chunk) == {
            "title": "t",
            "thinking": "T",
            "code": "x",
        }
        assert isinstance(out[2], ToolCallEnd)
        assert out[2].call_id == "call_1"

    def test_thought_signature_forwarded_on_start(self):
        """Gemini 3 attaches ``thought_signature`` bytes to the Part
        wrapping a function_call; the adapter must forward it onto
        :class:`ToolCallStart` so the emission can replay it."""
        sig = b"\x01\x02\x03opaque-signature"
        chunks = [
            _chunk(
                _fc_part(
                    id="call_1",
                    name="write_file",
                    args={"path": "x", "content": "y"},
                    thought_signature=sig,
                )
            )
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        starts = [e for e in out if isinstance(e, ToolCallStart)]
        assert len(starts) == 1
        assert starts[0].signature == sig

    def test_multiple_function_calls_in_one_chunk(self):
        chunks = [
            _chunk(
                _fc_part(id="call_a", name="write_file", args={"path": "x"}),
                _fc_part(id="call_b", name="python_action", args={"title": "t"}),
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
        chunks = [
            _chunk(_fc_part(id="call_1", name="python_action")),
            _chunk(_fc_part(id="call_1", name="python_action")),
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        assert len([e for e in out if isinstance(e, ToolCallStart)]) == 1
        assert len([e for e in out if isinstance(e, ToolCallEnd)]) == 1

    def test_missing_id_synthesized(self):
        chunks = [
            _chunk(
                _fc_part(id=None, name="python_action"),
                _fc_part(id=None, name="write_file"),
            )
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        ids = [e.call_id for e in out if isinstance(e, ToolCallStart)]
        assert len(ids) == 2
        assert len(set(ids)) == 2  # unique

    def test_usage_captured(self):
        um = SimpleNamespace(prompt_token_count=42, candidates_token_count=7)
        chunks = [_chunk(usage_metadata=um)]
        usage: dict = {}
        list(translate_gemini_stream_to_events(iter(chunks), usage_holder=usage))
        assert usage["input_tokens"] == 42
        assert usage["output_tokens"] == 7

    def test_chunks_without_function_calls_produce_nothing(self):
        chunks = [
            _chunk(),
            SimpleNamespace(candidates=[], usage_metadata=None),
        ]
        out = list(translate_gemini_stream_to_events(iter(chunks)))
        assert out == []


def _thought_part(
    *,
    signature: bytes | None = None,
    text: str | None = None,
    thought: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        function_call=None,
        thought_signature=signature,
        text=text,
        thought=thought,
    )


class TestThoughtPartCapture:
    """Gemini 3 ships signatures on Parts that have no function_call —
    a ``thought`` Part with ``thought_signature`` bytes that signs
    the subsequent function_calls.  Its docs require these to
    round-trip at the same position.  Capture them as
    :class:`ThinkingEmission`\\ s so the renderer can put them back."""

    def test_signed_thought_part_becomes_thinking_emission(self):
        from agex.agent.emissions import ThinkingEmission
        from agex.llm.core import EmissionsBuilder
        from agex.llm.formats.tool_use.parser import parse_tool_events

        sig = b"THOUGHT_SIG"
        chunks = [
            _chunk(
                _thought_part(signature=sig, text="planning the call", thought=True),
                _fc_part(id="call_1", name="python_action", args={"code": "x"}),
            )
        ]
        tool_events = translate_gemini_stream_to_events(iter(chunks))
        tokens = list(parse_tool_events(tool_events))
        builder = EmissionsBuilder()
        for t in tokens:
            builder.process_token(t)
        response = builder.build()

        # A ThinkingEmission precedes the PythonEmission.
        assert len(response.emissions) >= 2
        first = response.emissions[0]
        assert isinstance(first, ThinkingEmission)
        assert first.signature == sig
        assert first.text == "planning the call"

    def test_thought_part_replays_as_thought_part(self):
        """Signed ThinkingEmissions render as ``thinking`` blocks on
        the agent side; the Gemini translator then re-materializes
        them as ``Part(thought=True, thought_signature=..., text=...)``
        — the same shape Gemini originally delivered."""
        sig = b"THOUGHT_SIG"
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "text": "reasoning",
                        "signature": sig,
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "python_action",
                        "input": {"code": "x"},
                    },
                ],
            }
        ]
        out = translate_messages_to_gemini(msgs)
        parts = out[0]["parts"]
        assert parts[0] == {
            "thought": True,
            "text": "reasoning",
            "thought_signature": sig,
        }
        # Function call follows at its original position.
        assert "function_call" in parts[1]

    def test_unsigned_thought_part_dropped(self):
        """A thought part with neither signature nor text has nothing
        to round-trip; the adapter should drop it so we don't emit
        empty ThinkingEmissions."""
        from agex.llm.core import EmissionsBuilder
        from agex.llm.formats.tool_use.parser import parse_tool_events

        chunks = [
            _chunk(
                _thought_part(signature=None, text=None, thought=True),
                _fc_part(id="c", name="python_action", args={}),
            )
        ]
        tool_events = translate_gemini_stream_to_events(iter(chunks))
        tokens = list(parse_tool_events(tool_events))
        builder = EmissionsBuilder()
        for t in tokens:
            builder.process_token(t)
        response = builder.build()
        from agex.agent.emissions import ThinkingEmission

        assert not any(isinstance(em, ThinkingEmission) for em in response.emissions)


class TestSignatureFullRoundTrip:
    """The load-bearing scenario: stream carries a signature in, the
    emission stores it, and the renderer puts it back out on the
    function_call for the next turn's request.  If any link in that
    chain drops the bytes Gemini 3 will 400 us."""

    def test_signature_survives_stream_to_tool_use_block(self):
        from agex.agent.emissions import PythonEmission
        from agex.agent.events import ActionEvent
        from agex.llm.core import EmissionsBuilder
        from agex.llm.formats.tool_use.parser import parse_tool_events
        from agex.llm.formats.tool_use.renderer import render_events_as_tool_use

        sig_py = b"PY_SIG"
        sig_file = b"FILE_SIG"

        # Simulate two Gemini 3 function_calls, each with its own
        # signature on the wrapping Part.
        chunks = [
            _chunk(
                _fc_part(
                    id="call_file",
                    name="write_file",
                    args={"path": "/h.py", "content": "X=1"},
                    thought_signature=sig_file,
                ),
                _fc_part(
                    id="call_py",
                    name="python_action",
                    args={"title": "t", "thinking": "T", "code": "x"},
                    thought_signature=sig_py,
                ),
            )
        ]

        # Stream → parser → builder, mimicking the client path.
        tool_events = translate_gemini_stream_to_events(iter(chunks))
        tokens = list(parse_tool_events(tool_events))
        builder = EmissionsBuilder()
        for t in tokens:
            builder.process_token(t)
        response = builder.build()

        # Signatures landed on the right emissions.
        em_by_type = {type(em).__name__: em for em in response.emissions}
        assert em_by_type["FileWriteEmission"].signature == sig_file
        # Action-tool thinking may produce a leading ThinkingEmission; the
        # signature goes on the PythonEmission alongside the code.
        py = em_by_type["PythonEmission"]
        assert isinstance(py, PythonEmission)
        assert py.signature == sig_py

        # Now render that back as if we were building the next turn and
        # verify the signature rides on the tool_use blocks.
        event = ActionEvent(agent_name="a", emissions=list(response.emissions))
        rendered = render_events_as_tool_use([event])
        assistant = rendered[0]
        tool_uses = [b for b in assistant["content"] if b.get("type") == "tool_use"]
        assert len(tool_uses) == 2
        assert tool_uses[0]["signature"] == sig_file
        assert tool_uses[1]["signature"] == sig_py

        # Gemini translator then turns that into a Part whose
        # ``thought_signature`` lives next to ``function_call`` — the
        # final step that silences the 400.
        gemini_contents = translate_messages_to_gemini(rendered)
        model_turn = next(c for c in gemini_contents if c["role"] == "model")
        assert [p["thought_signature"] for p in model_turn["parts"]] == [
            sig_file,
            sig_py,
        ]
        # Sigs must live on the Part, not inside the FunctionCall.
        for p in model_turn["parts"]:
            assert "thought_signature" not in p["function_call"]


@pytest.mark.asyncio
async def test_async_stream_translator():
    async def gen():
        yield _chunk(_fc_part(id="c1", name="python_action", args={"title": "t"}))

    out = []
    async for e in atranslate_gemini_stream_to_events(gen()):
        out.append(e)
    assert [type(e).__name__ for e in out] == [
        "ToolCallStart",
        "ToolCallArgDelta",
        "ToolCallEnd",
    ]
