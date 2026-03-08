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
