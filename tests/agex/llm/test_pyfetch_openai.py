"""Tests for PyfetchOpenAI client — request formatting and stream parsing."""

import json

import pytest

from agex.llm.pyfetch_openai import PyfetchOpenAI, _format_message_for_openai
from agex.llm.sse import parse_sse_events

# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------


def test_format_text_message():
    msg = {"role": "user", "content": "hello"}
    assert _format_message_for_openai(msg) == msg


def test_format_multimodal_message():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "image_data": "abc123base64"},
        ],
    }
    result = _format_message_for_openai(msg)
    assert result["role"] == "user"
    assert len(result["content"]) == 2
    assert result["content"][0] == {"type": "text", "text": "describe this"}
    assert result["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123base64"},
    }


class TestCachePreservesToolUseFields:
    """Regression guard: the ``cache=True`` path used to drop
    ``tool_calls`` / ``tool_call_id`` from tool-use-shaped messages
    (emitted by ``translate_messages_to_openai``). Those fields MUST
    flow through to the API or the request is invalid."""

    def test_assistant_with_tool_calls_preserves_field(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "python_action",
                        "arguments": '{"x":1}',
                    },
                }
            ],
        }
        result = _format_message_for_openai(msg, cache=True)
        assert result["tool_calls"] == msg["tool_calls"]
        # content None must NOT get wrapped into an invalid text:None block.
        assert result["content"] is None

    def test_tool_role_preserves_tool_call_id_and_wraps_content(self):
        msg = {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"}
        result = _format_message_for_openai(msg, cache=True)
        assert result["tool_call_id"] == "toolu_1"
        # content wrapped to a single text block with cache_control.
        assert isinstance(result["content"], list)
        assert result["content"][0]["text"] == "ok"
        assert result["content"][0]["cache_control"]["type"] == "ephemeral"

    def test_user_string_wrap_preserves_role(self):
        msg = {"role": "user", "content": "hi"}
        result = _format_message_for_openai(msg, cache=True)
        assert result["role"] == "user"
        assert result["content"][0]["text"] == "hi"
        assert "cache_control" in result["content"][0]

    def test_multimodal_cache_goes_to_last_block(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "here:"},
                {"type": "image", "image_data": "BYTES"},
            ],
        }
        result = _format_message_for_openai(msg, cache=True)
        assert "cache_control" not in result["content"][0]
        assert "cache_control" in result["content"][1]

    def test_passthrough_when_no_cache_and_content_is_string(self):
        """XML-mode messages without cache should pass through unchanged."""
        msg = {"role": "user", "content": "hello"}
        assert _format_message_for_openai(msg) == msg

    def test_content_is_none_without_cache_pass_through(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "a", "type": "function", "function": {}}],
        }
        result = _format_message_for_openai(msg)
        # Shape preserved, extras preserved.
        assert result["content"] is None
        assert result["tool_calls"] == msg["tool_calls"]


# ---------------------------------------------------------------------------
# Client config
# ---------------------------------------------------------------------------


def test_dump_config():
    client = PyfetchOpenAI(
        model="anthropic/claude-sonnet-4",
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
    )
    config = client.dump_config()
    assert config["provider"] == "pyfetch_openai"
    assert config["model"] == "anthropic/claude-sonnet-4"
    assert config["base_url"] == "https://openrouter.ai/api/v1"
    assert config["temperature"] == 0.7
    assert "api_key" not in config  # Key should not be serialized


def test_default_base_url():
    client = PyfetchOpenAI(model="test", api_key="sk-test")
    assert client._base_url == "https://openrouter.ai/api/v1"


def test_base_url_trailing_slash_stripped():
    client = PyfetchOpenAI(
        model="test", api_key="sk-test", base_url="https://example.com/v1/"
    )
    assert client._base_url == "https://example.com/v1"


def test_sync_stream_raises():
    client = PyfetchOpenAI(model="test", api_key="sk-test")
    with pytest.raises(NotImplementedError, match="async"):
        client.complete_stream("system", [])


# ---------------------------------------------------------------------------
# SSE → TokenChunk integration
# ---------------------------------------------------------------------------


def _make_sse_chunk(content: str | None = None, usage: dict | None = None) -> str:
    """Build an SSE data line from an OpenAI-style chunk."""
    chunk: dict = {"choices": []}
    if content is not None:
        chunk["choices"] = [{"delta": {"content": content}}]
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


@pytest.mark.asyncio
async def test_sse_to_content_extraction():
    """Verify we extract content deltas from SSE chunks correctly."""

    async def fake_stream():
        yield _make_sse_chunk("<TITLE>Test</TITLE>")
        yield _make_sse_chunk("<THINKING>ok</THINKING>")
        yield _make_sse_chunk(usage={"prompt_tokens": 10, "completion_tokens": 5})
        yield "data: [DONE]\n\n"

    # Parse SSE events and extract content
    contents = []
    async for payload in parse_sse_events(fake_stream()):
        data = json.loads(payload)
        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                contents.append(content)

    assert contents == ["<TITLE>Test</TITLE>", "<THINKING>ok</THINKING>"]


@pytest.mark.asyncio
async def test_usage_extraction():
    """Verify we capture token usage from the final SSE chunk."""

    async def fake_stream():
        yield _make_sse_chunk("hello")
        yield _make_sse_chunk(usage={"prompt_tokens": 42, "completion_tokens": 17})
        yield "data: [DONE]\n\n"

    usage_holder: dict = {}
    async for payload in parse_sse_events(fake_stream()):
        data = json.loads(payload)
        usage = data.get("usage")
        if usage:
            usage_holder["input"] = usage.get("prompt_tokens")
            usage_holder["output"] = usage.get("completion_tokens")

    assert usage_holder == {"input": 42, "output": 17}
