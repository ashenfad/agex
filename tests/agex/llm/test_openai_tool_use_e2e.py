"""End-to-end tests for OpenAI clients in tool-use mode.

Mocks the OpenAI streaming API to return a canned sequence of
``ChatCompletionChunk``-shaped dicts and verifies that the client
dispatches to the tool-use path, yields TokenChunks, and produces a
valid :class:`LLMResponse` when run through :class:`EmissionsBuilder`.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agex.agent.emissions import FileEditEmission, FileWriteEmission
from agex.agent.events import TaskStartEvent
from agex.llm.core import EmissionsBuilder
from agex.llm.formats import ToolUseWireFormat
from agex.llm.openai_client import OpenAI
from agex.llm.pyfetch_openai import PyfetchOpenAI
from tests.agex._emissions import (
    make_action_event,
    response_code,
    response_file_actions,
    response_thinking,
    response_title,
)


def _mk_chunk(choices=None, usage=None):
    """Build a MagicMock chunk matching the SDK shape we consume."""
    chunk = MagicMock()
    chunk.usage = usage
    chunk.choices = choices or []
    # Translator normalizes via .model_dump(); provide a matching dict.
    chunk.model_dump = lambda: {
        "choices": [
            {
                "index": c["index"],
                "delta": c.get("delta", {}),
                "finish_reason": c.get("finish_reason"),
            }
            for c in (choices or [])
        ],
        "usage": usage,
    }
    return chunk


def _mk_tool_chunk(index, call_id=None, name=None, args=None, finish=None):
    tc: dict = {"index": index}
    fn = {}
    if call_id:
        tc["id"] = call_id
    if name:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        tc["function"] = fn
    return _mk_chunk(
        choices=[
            {
                "index": 0,
                "delta": {"tool_calls": [tc]},
                "finish_reason": finish,
            }
        ]
    )


def _mk_usage_chunk(prompt=10, completion=5):
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    # The SDK model_dump should have integer keys, not Mock
    chunk = _mk_chunk(choices=[], usage=usage)
    chunk.model_dump = lambda: {
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }
    return chunk


# ---------------------------------------------------------------------------
# Sync SDK client
# ---------------------------------------------------------------------------


class TestOpenAIToolUse:
    def test_python_action_round_trip(self):
        """Mocked stream emits a python_action tool call; output
        should populate LLMResponse.title/thinking/code."""
        args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})
        stream = [
            _mk_tool_chunk(0, call_id="call_a", name="python_action", args=""),
            _mk_tool_chunk(0, args=args_json),
            _mk_tool_chunk(0, args="", finish="tool_calls"),
            _mk_usage_chunk(prompt=12, completion=3),
        ]

        client = OpenAI(
            api_key="test", wire_format=ToolUseWireFormat(), use_responses=False
        )
        with patch.object(
            client.client.chat.completions, "create", return_value=iter(stream)
        ) as mock_create:
            system = "You are helpful."
            events = [
                TaskStartEvent(agent_name="a", task_name="t", inputs={}, message="go")
            ]
            tokens = list(client.complete_stream(system, events))

            # create() was called with tools= and tool_choice="required"
            # (tools are agex's API — the loop needs a tool call each
            # turn to make progress).
            call_kwargs = mock_create.call_args.kwargs
            assert "tools" in call_kwargs
            assert call_kwargs["tool_choice"] == "required"
            assert len(call_kwargs["tools"]) == 4
            assert {t["function"]["name"] for t in call_kwargs["tools"]} == {
                "python_action",
                "terminal_action",
                "write_file",
                "edit_file",
            }

        # Feed tokens through EmissionsBuilder to check end state.
        builder = EmissionsBuilder(agent_name="a")
        for t in tokens:
            builder.process_token(t)
        resp = builder.build()
        assert response_title(resp) == "t"
        assert response_thinking(resp) == "T"
        assert response_code(resp) == "print(1)"
        assert resp.input_tokens == 12
        assert resp.output_tokens == 3

    def test_write_file_plus_python(self):
        """Two parallel tool calls: write_file then python_action."""
        file_args = json.dumps({"path": "/a.py", "content": "X = 1"})
        py_args = json.dumps({"title": "t", "thinking": "T", "code": "import a"})
        stream = [
            _mk_tool_chunk(0, call_id="call_f", name="write_file", args=""),
            _mk_tool_chunk(0, args=file_args),
            _mk_tool_chunk(1, call_id="call_p", name="python_action", args=""),
            _mk_tool_chunk(1, args=py_args),
            _mk_tool_chunk(0, args="", finish="tool_calls"),
            _mk_usage_chunk(),
        ]

        client = OpenAI(
            api_key="test", wire_format=ToolUseWireFormat(), use_responses=False
        )
        with patch.object(
            client.client.chat.completions, "create", return_value=iter(stream)
        ):
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

    def test_edit_file_replace_with_match_all(self):
        edit_args = json.dumps(
            {
                "path": "/b.py",
                "search": "anchor",
                "replace": "anchor + added",
                "match_all": True,
            }
        )
        py_args = json.dumps({"title": "t", "thinking": "T", "code": "pass"})
        stream = [
            _mk_tool_chunk(0, call_id="call_e", name="edit_file", args=edit_args),
            _mk_tool_chunk(1, call_id="call_p", name="python_action", args=py_args),
            _mk_tool_chunk(0, args="", finish="tool_calls"),
            _mk_usage_chunk(),
        ]

        client = OpenAI(
            api_key="test", wire_format=ToolUseWireFormat(), use_responses=False
        )
        with patch.object(
            client.client.chat.completions, "create", return_value=iter(stream)
        ):
            tokens = list(client.complete_stream("sys", []))

        builder = EmissionsBuilder(agent_name="a")
        for t in tokens:
            builder.process_token(t)
        resp = builder.build()

        assert len(response_file_actions(resp)) == 1
        ea = response_file_actions(resp)[0]
        assert isinstance(ea, FileEditEmission)
        assert ea.path == "/b.py"
        assert ea.search == "anchor"
        assert ea.content == "anchor + added"
        assert ea.match_all is True


# ---------------------------------------------------------------------------
# Async pyfetch client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pyfetch_openai_tool_use():
    """PyfetchOpenAI in tool-use mode: mock fetch_stream with canned SSE."""
    args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})

    sse_lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            }
        )
        + "\n\n",
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

    # Rely on the client's default wire_format (native_thinking=True
    # after the OpenRouter-reasoning migration) so this test also
    # pins the default ``body["reasoning"]`` injection.
    client = PyfetchOpenAI(
        model="anthropic/claude-sonnet-4",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
    )

    tokens = []
    async for t in client.acomplete_stream("sys", []):
        tokens.append(t)

    body = fake_adapter.fetch_stream.call_args.kwargs["body"]
    assert "tools" in body
    # With reasoning on, ``tool_choice`` drops from ``required`` to
    # ``auto`` — Anthropic's extended thinking is incompatible with a
    # forced tool call, and OpenRouter forwards the flag verbatim.
    assert body["tool_choice"] == "auto"
    # Default ``native_thinking=True`` should inject OpenRouter's
    # unified reasoning block.  For ``anthropic/*`` models we use the
    # Anthropic-native budget shape (``max_tokens``) rather than
    # OpenAI-native ``effort``, since OpenRouter's effort→budget
    # conversion is unreliable on the Anthropic route.
    assert body["reasoning"] == {"enabled": True, "max_tokens": 2048}
    # OpenRouter-only capability filter — pin to Anthropic direct so
    # Bedrock/Vertex don't silently drop reasoning.
    assert body["provider"] == {"order": ["Anthropic"], "allow_fallbacks": False}

    # Round-trip through EmissionsBuilder.
    builder = EmissionsBuilder(agent_name="a")
    for t in tokens:
        builder.process_token(t)
    resp = builder.build()
    assert response_title(resp) == "t"
    assert response_code(resp) == "print(1)"
    assert resp.input_tokens == 20
    assert resp.output_tokens == 5


@pytest.mark.asyncio
async def test_pyfetch_openai_native_thinking_off_skips_reasoning_block():
    """When the caller opts out of native thinking explicitly,
    ``body["reasoning"]`` should not be set — keeps compatibility with
    routes that don't understand OpenRouter's reasoning extension.
    """
    from agex.llm.formats import ToolUseWireFormat

    args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})
    sse_lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\n",
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

    client = PyfetchOpenAI(
        model="anthropic/claude-sonnet-4",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
        wire_format=ToolUseWireFormat(native_thinking=False),
    )
    async for _ in client.acomplete_stream("sys", []):
        pass

    body = fake_adapter.fetch_stream.call_args.kwargs["body"]
    assert "reasoning" not in body
    # With reasoning off, the loop-progression safety net kicks back
    # in: tool_choice stays ``required`` so the model can't stall
    # with plain assistant text.
    assert body["tool_choice"] == "required"
    # No reasoning kwarg → no OpenRouter ``provider`` capability
    # pinning either (it's only relevant when we're actually asking
    # for reasoning_details back).
    assert "provider" not in body


@pytest.mark.asyncio
async def test_pyfetch_openai_non_anthropic_openrouter_uses_require_parameters():
    """Non-Anthropic OpenRouter models get ``require_parameters=True``
    rather than the Anthropic-pinning ``order`` filter.  Also verifies
    the reasoning shape falls through to OpenAI-native ``effort``
    (not the Anthropic-native ``max_tokens`` budget)."""
    args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})
    sse_lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\n",
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

    client = PyfetchOpenAI(
        # x-ai/grok doesn't match the anthropic/google-prefix branch,
        # so the client should use effort + require_parameters.
        model="x-ai/grok-2",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
    )
    async for _ in client.acomplete_stream("sys", []):
        pass

    body = fake_adapter.fetch_stream.call_args.kwargs["body"]
    assert body["reasoning"] == {"enabled": True, "effort": "medium"}
    assert body["provider"] == {"require_parameters": True}


@pytest.mark.asyncio
async def test_pyfetch_openai_non_openrouter_base_url_omits_provider():
    """The ``provider`` capability filter is an OpenRouter-specific
    field (not part of the OpenAI API).  When the client is pointed
    at a non-OpenRouter endpoint, we must not inject it — some
    OpenAI-compatible servers 400 on unknown top-level fields."""
    args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})
    sse_lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\n",
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

    client = PyfetchOpenAI(
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        fetch_adapter=fake_adapter,
    )
    async for _ in client.acomplete_stream("sys", []):
        pass

    body = fake_adapter.fetch_stream.call_args.kwargs["body"]
    # Reasoning injection still runs (native_thinking default is True)
    # — just the OpenRouter-specific ``provider`` hint is skipped.
    assert body["reasoning"] == {"enabled": True, "effort": "medium"}
    assert "provider" not in body


@pytest.mark.asyncio
async def test_pyfetch_openai_user_reasoning_kwarg_wins():
    """Caller-supplied ``reasoning=`` overrides the default
    injection — users running custom reasoning budgets or disabling
    on specific routes should stay in control."""
    args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})
    sse_lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\n",
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

    client = PyfetchOpenAI(
        model="anthropic/claude-sonnet-4",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
        reasoning={"effort": "high", "max_tokens": 4000},
    )
    async for _ in client.acomplete_stream("sys", []):
        pass

    body = fake_adapter.fetch_stream.call_args.kwargs["body"]
    assert body["reasoning"] == {"effort": "high", "max_tokens": 4000}


@pytest.mark.asyncio
async def test_pyfetch_openai_responses_path():
    """PyfetchOpenAI on a GPT-5-family model hits ``/v1/responses`` with
    the Responses-shaped body (flat tool schema, ``input`` items,
    ``reasoning`` block, encrypted-content include) and parses
    responses-shape SSE events correctly.
    """
    fc_args = json.dumps({"title": "t", "code": "pass"})
    sse_lines = [
        "data: "
        + json.dumps(
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
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": fc_args,
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "python_action",
                    "arguments": fc_args,
                },
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                    }
                },
            }
        )
        + "\n\n",
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

    client = PyfetchOpenAI(
        model="openai/gpt-5-mini",  # triggers Responses dispatch
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        fetch_adapter=fake_adapter,
        wire_format=ToolUseWireFormat(),
    )
    assert client._use_responses is True

    tokens = []
    async for t in client.acomplete_stream("sys", []):
        tokens.append(t)

    # Sent to the Responses endpoint with the Responses-shaped body.
    call = fake_adapter.fetch_stream.call_args
    url = call.args[0] if call.args else call.kwargs.get("url")
    assert url.endswith("/responses")
    body = call.kwargs["body"]
    assert "input" in body and "messages" not in body
    assert body["tool_choice"] == "required"
    assert body["store"] is False
    assert "reasoning.encrypted_content" in body["include"]
    assert body["instructions"].startswith("sys")
    # Flat tool shape: no nested ``function`` wrapper.
    tools = body["tools"]
    assert tools and all("function" not in t for t in tools)

    # Stream parsed into a clean LLMResponse.
    builder = EmissionsBuilder(agent_name="a")
    for t in tokens:
        builder.process_token(t)
    resp = builder.build()
    assert response_code(resp) == "pass"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 20


@pytest.mark.asyncio
async def test_pyfetch_openai_logs_cache_diagnostics(capsys):
    """Per-request the client emits one line tagged
    ``[agex.llm.cache]`` carrying the prefix hashes and the cached/
    prompt token counts the provider reported. Lets us diff
    consecutive turns to spot prefix drift vs. provider-side misses.
    The line goes to stdout (not ``logging``) so Pyodide routes it
    to the browser console."""
    args_json = json.dumps({"title": "t", "thinking": "T", "code": "x"})
    sse_lines = [
        "data: "
        + json.dumps(
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
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\n",
        # Final usage chunk, including OpenRouter's cached-prompt detail.
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 16967,
                    "completion_tokens": 469,
                    "prompt_tokens_details": {"cached_tokens": 11915},
                },
            }
        )
        + "\n\n",
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

    client = PyfetchOpenAI(
        model="anthropic/claude-sonnet-4",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
        wire_format=ToolUseWireFormat(),
    )

    async for _ in client.acomplete_stream("sys", []):
        pass

    captured = capsys.readouterr().out
    cache_lines = [ln for ln in captured.splitlines() if "[agex.llm.cache]" in ln]
    assert len(cache_lines) == 1
    line = cache_lines[0]
    # The diagnostic must surface the actually-reported cache hit so
    # consecutive turns can be eyeballed.
    assert "cached_tokens=11915" in line
    assert "prompt_tokens=16967" in line
    # And the per-position hashes so two requests can be diff'd to find
    # where (if anywhere) drift starts.
    assert "sys_hash=" in line
    assert "prefix_hash=" in line


@pytest.mark.asyncio
async def test_pyfetch_openai_logs_provider_and_cache_write(capsys):
    """When OpenRouter returns a top-level ``provider`` field and
    ``cache_write_tokens`` / ``cache_discount`` in the usage object,
    the diagnostic must surface them. Lets us spot when sticky routing
    bounces between upstream providers (the leading hypothesis for
    intermittent cache hits with stable prefixes)."""
    args_json = json.dumps({"title": "t", "thinking": "T", "code": "x"})
    sse_lines = [
        "data: "
        + json.dumps(
            {
                # OpenRouter typically attaches `provider` to the first
                # chunk of the stream.
                "provider": "Google Vertex",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 12000,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {
                        "cached_tokens": 8000,
                        "cache_write_tokens": 3500,
                    },
                },
                "cache_discount": 0.42,
            }
        )
        + "\n\n",
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

    client = PyfetchOpenAI(
        model="google/gemini-2.5-pro",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
        wire_format=ToolUseWireFormat(),
    )

    async for _ in client.acomplete_stream("sys", []):
        pass

    captured = capsys.readouterr().out
    cache_lines = [ln for ln in captured.splitlines() if "[agex.llm.cache]" in ln]
    assert len(cache_lines) == 1
    line = cache_lines[0]
    assert "cached_tokens=8000" in line
    assert "cache_write_tokens=3500" in line
    assert "cache_discount=0.42" in line
    assert "provider=Google Vertex" in line


@pytest.mark.asyncio
async def test_pyfetch_openai_cache_marker_lands_on_cacheable_block():
    """Regression: in tool-use mode the prior cache_idx=len-2 landed
    on an assistant-with-tool_calls message whose content is None,
    and our cache helper has no content block to attach the marker
    to — the breakpoint silently disappeared every turn so OpenRouter
    only cached the system prompt. Must land on a block that actually
    receives a cache_control marker."""
    from agex.agent.events import OutputEvent

    args_json = json.dumps({"title": "t", "thinking": "T", "code": "x"})
    sse_lines = [
        "data: "
        + json.dumps(
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
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\n",
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

    client = PyfetchOpenAI(
        model="anthropic/claude-sonnet-4",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
        wire_format=ToolUseWireFormat(),
    )

    # Multi-turn-ish history: a TaskStart and one prior python_action
    # round whose tool_result is the LAST message (this is the typical
    # shape between turns).
    events = [
        TaskStartEvent(agent_name="a", task_name="t", inputs={}, message="do work"),
        make_action_event(
            agent_name="a",
            title="t1",
            thinking="T1",
            code="print('hi'); task_continue()",
        ),
        OutputEvent(agent_name="a", parts=["hi"]),
    ]

    tokens = []
    async for t in client.acomplete_stream("sys", events):
        tokens.append(t)

    body = fake_adapter.fetch_stream.call_args.kwargs["body"]
    msgs = body["messages"]

    # System message should be cached.
    sys_msg = msgs[0]
    sys_blocks = sys_msg["content"] if isinstance(sys_msg["content"], list) else []
    assert any(isinstance(b, dict) and b.get("cache_control") for b in sys_blocks), (
        "system message should carry cache_control"
    )

    # AT LEAST ONE non-system message should also carry cache_control,
    # otherwise we'd only cache the system prompt and miss the entire
    # conversation history.
    def _has_cache_control(msg):
        c = msg.get("content")
        if isinstance(c, list):
            return any(isinstance(b, dict) and b.get("cache_control") for b in c)
        return False

    cached_non_system = [m for m in msgs[1:] if _has_cache_control(m)]
    assert len(cached_non_system) >= 1, (
        "no non-system message has cache_control — cache breakpoint lost"
    )
    """PyfetchOpenAI in tool-use mode: mock fetch_stream with canned SSE."""
    args_json = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})

    sse_lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "python_action",
                                        "arguments": args_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            }
        )
        + "\n\n",
        "data: [DONE]\n\n",
    ]

    # Fake fetch_stream: parse_sse_events consumes its return value
    # directly as an async iterator of str chunks.
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

    client = PyfetchOpenAI(
        model="gpt-4",
        api_key="sk-test",
        fetch_adapter=fake_adapter,
        wire_format=ToolUseWireFormat(),
    )

    tokens = []
    async for t in client.acomplete_stream("sys", []):
        tokens.append(t)

    # Verify body passed to fetch_stream includes tools.
    call_kwargs = fake_adapter.fetch_stream.call_args.kwargs
    body = call_kwargs["body"]
    assert "tools" in body
    assert body["tool_choice"] == "required"

    # Round-trip through EmissionsBuilder.
    builder = EmissionsBuilder(agent_name="a")
    for t in tokens:
        builder.process_token(t)
    resp = builder.build()
    assert response_title(resp) == "t"
    assert response_code(resp) == "print(1)"
    assert resp.input_tokens == 20
    assert resp.output_tokens == 5
