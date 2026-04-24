"""End-to-end tests for Gemini in tool-use mode.

Mocks ``generate_content_stream`` to return a canned sequence of chunks
whose ``function_calls`` property carries ``FunctionCall``-shaped objects,
then verifies the client dispatches to the tool-use path, yields
TokenChunks, and produces a correct :class:`LLMResponse` through
:class:`EmissionsBuilder`.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agex.agent.emissions import FileEditEmission, FileWriteEmission
from agex.llm.core import EmissionsBuilder
from agex.llm.formats import ToolUseWireFormat
from agex.llm.gemini_client import Gemini
from tests.agex._emissions import (
    response_code,
    response_file_actions,
    response_thinking,
    response_title,
)


def _fc(id_, name, args, *, signature=None):
    """Build a mock Gemini Part wrapping a function_call.

    Gemini 3 attaches ``thought_signature`` bytes to the Part; the
    adapter walks ``candidates[*].content.parts`` to pick them up.
    """
    return SimpleNamespace(
        function_call=SimpleNamespace(id=id_, name=name, args=args),
        thought_signature=signature,
    )


def _chunk(function_calls=None, usage=None):
    """Mock a streamed chunk: wrap parts in a single candidate's content."""
    parts = function_calls or []
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))],
        usage_metadata=usage,
    )


def _usage(prompt, completion):
    return SimpleNamespace(prompt_token_count=prompt, candidates_token_count=completion)


class TestGeminiToolUse:
    def test_python_action_round_trip(self):
        with patch("google.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content_stream.return_value = [
                _chunk(
                    function_calls=[
                        _fc(
                            "call_1",
                            "python_action",
                            {"title": "t", "thinking": "T", "code": "print(1)"},
                        )
                    ],
                    usage=_usage(20, 3),
                )
            ]

            client = Gemini(wire_format=ToolUseWireFormat())
            tokens = list(client.complete_stream("sys", []))

            # generate_content_stream was called with a config that includes
            # our function-declaration tool.
            kwargs = mock_models.generate_content_stream.call_args.kwargs
            config = kwargs["config"]
            tool_names: list[str] = []
            for t in config.tools or []:
                decls = getattr(t, "function_declarations", None)
                if decls:
                    tool_names.extend(d.name for d in decls)
            assert {
                "python_action",
                "terminal_action",
                "write_file",
                "edit_file",
            }.issubset(set(tool_names))

            # Messages (contents) are translated Gemini dicts, not legacy
            # text-path Content objects with XML prefill.
            contents = kwargs["contents"]
            # No trailing "<TITLE>" prefill model turn in tool-use mode.
            for c in contents:
                role = c["role"] if isinstance(c, dict) else c.role
                if role == "model":
                    parts = c["parts"] if isinstance(c, dict) else c.parts
                    texts = [
                        p.get("text", "") if isinstance(p, dict) else (p.text or "")
                        for p in parts
                    ]
                    assert not any("<TITLE>" in t for t in texts)

            # Round-trip to LLMResponse.
            builder = EmissionsBuilder(agent_name="a")
            for t in tokens:
                builder.process_token(t)
            resp = builder.build()
            assert response_title(resp) == "t"
            assert response_thinking(resp) == "T"
            assert response_code(resp) == "print(1)"
            assert resp.input_tokens == 20
            assert resp.output_tokens == 3

    def test_write_file_plus_python(self):
        with patch("google.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content_stream.return_value = [
                _chunk(
                    function_calls=[
                        _fc(
                            "call_f",
                            "write_file",
                            {"path": "/a.py", "content": "X = 1"},
                        ),
                        _fc(
                            "call_p",
                            "python_action",
                            {"title": "t", "thinking": "T", "code": "import a"},
                        ),
                    ]
                )
            ]

            client = Gemini(wire_format=ToolUseWireFormat())
            tokens = list(client.complete_stream("sys", []))
            builder = EmissionsBuilder(agent_name="a")
            for t in tokens:
                builder.process_token(t)
            resp = builder.build()

            assert response_code(resp) == "import a"
            assert len(response_file_actions(resp)) == 1
            fa = response_file_actions(resp)[0]
            assert isinstance(fa, FileWriteEmission)
            assert fa.path == "/a.py"
            assert fa.content == "X = 1"

    def test_edit_file(self):
        with patch("google.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content_stream.return_value = [
                _chunk(
                    function_calls=[
                        _fc(
                            "call_e",
                            "edit_file",
                            {
                                "path": "/b.py",
                                "search": "x",
                                "replace": "y",
                            },
                        ),
                        _fc(
                            "call_p",
                            "python_action",
                            {"title": "t", "thinking": "T", "code": "pass"},
                        ),
                    ]
                )
            ]

            client = Gemini(wire_format=ToolUseWireFormat())
            tokens = list(client.complete_stream("sys", []))
            builder = EmissionsBuilder(agent_name="a")
            for t in tokens:
                builder.process_token(t)
            resp = builder.build()

            assert len(response_file_actions(resp)) == 1
            ea = response_file_actions(resp)[0]
            assert isinstance(ea, FileEditEmission)
            assert ea.path == "/b.py"
            assert ea.content == "y"

    def test_grounding_tools_coexist_with_function_declarations(self):
        """When google_search or url_context is enabled, the tools list
        sent to Gemini should contain BOTH the grounding tool and our
        function_declarations tool."""
        with patch("google.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content_stream.return_value = [
                _chunk(
                    function_calls=[
                        _fc(
                            "c1",
                            "python_action",
                            {"title": "t", "thinking": "T", "code": "pass"},
                        )
                    ]
                )
            ]

            client = Gemini(wire_format=ToolUseWireFormat(), google_search=True)
            list(client.complete_stream("sys", []))

            kwargs = mock_models.generate_content_stream.call_args.kwargs
            tools = kwargs["config"].tools
            has_search = any(
                getattr(t, "google_search", None) is not None for t in tools
            )
            has_fns = any(getattr(t, "function_declarations", None) for t in tools)
            assert has_search
            assert has_fns


@pytest.mark.asyncio
async def test_gemini_async_tool_use():
    with patch("google.genai.Client") as MockClient:
        mock_aio_models = MockClient.return_value.aio.models

        async def fake_stream():
            yield _chunk(
                function_calls=[
                    _fc(
                        "call_1",
                        "python_action",
                        {
                            "title": "t",
                            "thinking": "T",
                            "code": "print(1)",
                        },
                    )
                ],
                usage=_usage(30, 9),
            )

        # aio.models.generate_content_stream is an async function
        # returning an async iterator.
        async def ret_stream(**_):
            return fake_stream()

        mock_aio_models.generate_content_stream = ret_stream

        client = Gemini(wire_format=ToolUseWireFormat())
        tokens = []
        async for t in client.acomplete_stream("sys", []):
            tokens.append(t)

        builder = EmissionsBuilder(agent_name="a")
        for t in tokens:
            builder.process_token(t)
        resp = builder.build()
        assert response_title(resp) == "t"
        assert response_code(resp) == "print(1)"
        assert resp.input_tokens == 30
        assert resp.output_tokens == 9
