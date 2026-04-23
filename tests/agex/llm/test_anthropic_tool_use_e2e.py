"""End-to-end tests for Anthropic clients in tool-use mode.

Mocks the Anthropic streaming API to return canned event sequences and
verifies that each client dispatches to the tool-use path, yields
TokenChunks, and produces a correct :class:`LLMResponse` when run
through :class:`ResponseBuilder`.
"""

import json
from unittest.mock import MagicMock

import pytest

from agex.agent.emissions import FileEditEmission, FileWriteEmission
from agex.llm.anthropic_client import Anthropic
from agex.llm.core import ResponseBuilder
from agex.llm.formats import ToolUseWireFormat, XmlWireFormat
from agex.llm.pyfetch_anthropic import PyfetchAnthropic
from tests.agex._emissions import (
    response_code,
    response_file_actions,
    response_thinking,
    response_title,
)


def _mk_event(d: dict):
    """Wrap a dict in a fake ``MessageStreamEvent`` — the SDK normalizer
    calls ``.model_dump()``."""
    ev = MagicMock()
    ev.model_dump = lambda: d
    return ev


def _mk_cm(stream_events: list):
    """Return a MagicMock context manager that iterates over events."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=iter(stream_events))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestAnthropicToolUse:
    def test_python_action_round_trip(self):
        args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})
        events = [
            _mk_event(
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 20, "output_tokens": 0}},
                }
            ),
            _mk_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "python_action",
                    },
                }
            ),
            _mk_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": args_json,
                    },
                }
            ),
            _mk_event({"type": "content_block_stop", "index": 0}),
            _mk_event(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 7},
                }
            ),
            _mk_event({"type": "message_stop"}),
        ]

        client = Anthropic(api_key="test", wire_format=ToolUseWireFormat())
        client.client = MagicMock()
        client.client.messages.stream = MagicMock(return_value=_mk_cm(events))

        tokens = list(client.complete_stream("sys", []))

        # stream() was called with tools=...
        call_kwargs = client.client.messages.stream.call_args.kwargs
        assert "tools" in call_kwargs
        tools = call_kwargs["tools"]
        assert {t["name"] for t in tools} == {
            "python_action",
            "terminal_action",
            "write_file",
            "edit_file",
        }
        # Each tool uses input_schema (not parameters).
        assert all("input_schema" in t for t in tools)

        # Round-trip to LLMResponse.
        builder = ResponseBuilder(agent_name="a")
        for t in tokens:
            builder.process_token(t)
        resp = builder.build()
        assert response_title(resp) == "t"
        assert response_thinking(resp) == "T"
        assert response_code(resp) == "print(1)"
        assert resp.input_tokens == 20
        assert resp.output_tokens == 7

    def test_write_file_plus_python(self):
        file_args = json.dumps({"path": "/a.py", "content": "X = 1"})
        py_args = json.dumps({"title": "t", "thinking": "T", "code": "import a"})
        events = [
            _mk_event({"type": "message_start", "message": {"usage": {}}}),
            _mk_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_f",
                        "name": "write_file",
                    },
                }
            ),
            _mk_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": file_args,
                    },
                }
            ),
            _mk_event({"type": "content_block_stop", "index": 0}),
            _mk_event(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_p",
                        "name": "python_action",
                    },
                }
            ),
            _mk_event(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": py_args,
                    },
                }
            ),
            _mk_event({"type": "content_block_stop", "index": 1}),
            _mk_event({"type": "message_stop"}),
        ]

        client = Anthropic(api_key="test", wire_format=ToolUseWireFormat())
        client.client = MagicMock()
        client.client.messages.stream = MagicMock(return_value=_mk_cm(events))

        tokens = list(client.complete_stream("sys", []))
        builder = ResponseBuilder(agent_name="a")
        for t in tokens:
            builder.process_token(t)
        resp = builder.build()

        assert response_code(resp) == "import a"
        assert len(response_file_actions(resp)) == 1
        fa = response_file_actions(resp)[0]
        assert isinstance(fa, FileWriteEmission)
        assert fa.path == "/a.py"
        assert fa.content == "X = 1"

    def test_edit_file_insert_before(self):
        edit_args = json.dumps(
            {
                "path": "/b.py",
                "search": "anchor",
                "insert_before": "added",
            }
        )
        py_args = json.dumps({"title": "t", "thinking": "T", "code": "pass"})
        events = [
            _mk_event({"type": "message_start", "message": {"usage": {}}}),
            _mk_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_e",
                        "name": "edit_file",
                    },
                }
            ),
            _mk_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": edit_args,
                    },
                }
            ),
            _mk_event({"type": "content_block_stop", "index": 0}),
            _mk_event(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_p",
                        "name": "python_action",
                    },
                }
            ),
            _mk_event(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": py_args,
                    },
                }
            ),
            _mk_event({"type": "content_block_stop", "index": 1}),
            _mk_event({"type": "message_stop"}),
        ]

        client = Anthropic(api_key="test", wire_format=ToolUseWireFormat())
        client.client = MagicMock()
        client.client.messages.stream = MagicMock(return_value=_mk_cm(events))

        tokens = list(client.complete_stream("sys", []))
        builder = ResponseBuilder(agent_name="a")
        for t in tokens:
            builder.process_token(t)
        resp = builder.build()

        assert len(response_file_actions(resp)) == 1
        ea = response_file_actions(resp)[0]
        assert isinstance(ea, FileEditEmission)
        assert ea.path == "/b.py"
        assert ea.operation == "insert-before"
        assert ea.content == "added"

    def test_xml_format_still_works(self):
        """Explicit XmlWireFormat path — tool-use API params should NOT
        be in the call."""

        class FakeTextStream:
            def __init__(self, texts):
                self._texts = texts

            def __iter__(self):
                return iter(self._texts)

        cm = MagicMock()
        cm.__enter__ = MagicMock(
            return_value=MagicMock(
                text_stream=FakeTextStream(
                    ["<THINKING>T</THINKING><PYTHON>pass</PYTHON>"]
                ),
                get_final_message=MagicMock(
                    return_value=MagicMock(
                        usage=MagicMock(input_tokens=5, output_tokens=2)
                    )
                ),
            )
        )
        cm.__exit__ = MagicMock(return_value=False)

        client = Anthropic(api_key="test", wire_format=XmlWireFormat())
        client.client = MagicMock()
        client.client.messages.stream = MagicMock(return_value=cm)

        list(client.complete_stream("sys", []))
        call_kwargs = client.client.messages.stream.call_args.kwargs
        assert "tools" not in call_kwargs


@pytest.mark.asyncio
async def test_pyfetch_anthropic_tool_use():
    """PyfetchAnthropic in tool-use mode: mock fetch_stream with canned SSE."""
    args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})

    def sse(payload):
        return "data: " + json.dumps(payload) + "\n\n"

    sse_lines = [
        sse(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 30, "output_tokens": 0}},
            }
        ),
        sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "python_action",
                },
            }
        ),
        sse(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": args_json,
                },
            }
        ),
        sse({"type": "content_block_stop", "index": 0}),
        sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 9},
            }
        ),
        sse({"type": "message_stop"}),
        "data: [DONE]\n\n",
    ]

    class FakeStream:
        def __init__(self, lines):
            self._lines = lines

        def __aiter__(self):
            async def gen():
                for line in self._lines:
                    yield line

            return gen()

    fake_adapter = MagicMock()
    fake_adapter.fetch_stream = MagicMock(return_value=FakeStream(sse_lines))

    client = PyfetchAnthropic(
        model="claude-sonnet-4-5",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
        wire_format=ToolUseWireFormat(),
    )

    tokens = []
    async for t in client.acomplete_stream("sys", []):
        tokens.append(t)

    call_kwargs = fake_adapter.fetch_stream.call_args.kwargs
    body = call_kwargs["body"]
    assert "tools" in body
    assert all("input_schema" in t for t in body["tools"])

    builder = ResponseBuilder(agent_name="a")
    for t in tokens:
        builder.process_token(t)
    resp = builder.build()
    assert response_title(resp) == "t"
    assert response_code(resp) == "print(1)"
    assert resp.input_tokens == 30
    assert resp.output_tokens == 9
