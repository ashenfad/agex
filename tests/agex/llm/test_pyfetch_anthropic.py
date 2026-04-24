"""Tests for PyfetchAnthropic client — request formatting and stream parsing."""

import json

import pytest

from agex.llm.pyfetch_anthropic import (
    PyfetchAnthropic,
    _format_message_for_anthropic,
)
from agex.llm.sse import parse_sse_events

# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------


def test_format_text_message():
    msg = {"role": "user", "content": "hello"}
    result = _format_message_for_anthropic(msg)
    assert result == {
        "role": "user",
        "content": [{"type": "text", "text": "hello"}],
    }


def test_format_multimodal_message():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "image_data": "abc123base64"},
        ],
    }
    result = _format_message_for_anthropic(msg)
    assert result["role"] == "user"
    assert len(result["content"]) == 2
    assert result["content"][0] == {"type": "text", "text": "describe this"}
    assert result["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "abc123base64",
        },
    }


def test_format_with_cache():
    msg = {"role": "user", "content": "hello"}
    result = _format_message_for_anthropic(msg, cache=True)
    assert result["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


# ---------------------------------------------------------------------------
# Client config
# ---------------------------------------------------------------------------


def test_dump_config():
    client = PyfetchAnthropic(
        model="claude-sonnet-4-5",
        api_key="sk-ant-test",
        temperature=0.7,
    )
    config = client.dump_config()
    assert config["provider"] == "pyfetch_anthropic"
    assert config["model"] == "claude-sonnet-4-5"
    assert config["base_url"] == "https://api.anthropic.com/v1"
    assert config["temperature"] == 0.7
    assert "api_key" not in config  # Key should not be serialized


def test_default_base_url():
    client = PyfetchAnthropic(model="test", api_key="sk-ant-test")
    assert client._base_url == "https://api.anthropic.com/v1"


def test_base_url_trailing_slash_stripped():
    client = PyfetchAnthropic(
        model="test", api_key="sk-ant-test", base_url="https://example.com/v1/"
    )
    assert client._base_url == "https://example.com/v1"


def test_headers():
    client = PyfetchAnthropic(model="test", api_key="sk-ant-test")
    h = client._headers()
    assert h["x-api-key"] == "sk-ant-test"
    assert h["anthropic-version"] == "2023-06-01"
    assert h["anthropic-dangerous-direct-browser-access"] == "true"
    assert h["Content-Type"] == "application/json"


def test_sync_stream_raises():
    client = PyfetchAnthropic(model="test", api_key="sk-ant-test")
    with pytest.raises(NotImplementedError, match="async"):
        client.complete_stream("system", [])


# ---------------------------------------------------------------------------
# tool_choice + extended-thinking compatibility
# ---------------------------------------------------------------------------


class TestEnsureToolChoiceAny:
    """Unit coverage for the SDK-parity helper that forces tool
    calling when extended thinking isn't in play.

    The rule is three-way: user-supplied ``tool_choice`` always wins;
    presence of ``thinking`` skips the force (Anthropic rejects
    ``tool_choice != auto`` with extended thinking enabled); otherwise
    default to ``{"type": "any"}`` so the model can't stall the agent
    loop by emitting plain assistant text.
    """

    def test_forces_any_when_no_thinking(self):
        from agex.llm.pyfetch_anthropic import _ensure_tool_choice_any

        out = _ensure_tool_choice_any({"max_tokens": 1024})
        assert out["tool_choice"] == {"type": "any"}
        assert out["max_tokens"] == 1024  # unchanged

    def test_skips_when_thinking_present(self):
        """Extended thinking + forced tool_choice is an API 400 —
        leave tool_choice unset so it defaults to ``auto`` server-side."""
        from agex.llm.pyfetch_anthropic import _ensure_tool_choice_any

        kwargs = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        out = _ensure_tool_choice_any(kwargs)
        assert "tool_choice" not in out
        # Original kwargs should round-trip untouched.
        assert out["thinking"] == kwargs["thinking"]

    def test_user_tool_choice_wins_over_default(self):
        from agex.llm.pyfetch_anthropic import _ensure_tool_choice_any

        out = _ensure_tool_choice_any({"tool_choice": "auto"})
        assert out["tool_choice"] == "auto"

    def test_user_tool_choice_wins_even_with_thinking(self):
        """If a caller explicitly sets ``tool_choice`` alongside
        ``thinking``, they own the incompatibility — don't second-
        guess them."""
        from agex.llm.pyfetch_anthropic import _ensure_tool_choice_any

        out = _ensure_tool_choice_any(
            {
                "tool_choice": {"type": "tool", "name": "python_action"},
                "thinking": {"type": "enabled", "budget_tokens": 2048},
            }
        )
        assert out["tool_choice"] == {"type": "tool", "name": "python_action"}


# ---------------------------------------------------------------------------
# SSE → content extraction
# ---------------------------------------------------------------------------


def _sse(event_type: str, **fields) -> str:
    """Build an Anthropic-style SSE event data line."""
    data = {"type": event_type, **fields}
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@pytest.mark.asyncio
async def test_sse_content_block_delta_extraction():
    """Verify we extract text from content_block_delta events."""

    async def fake_stream():
        yield _sse(
            "message_start",
            message={"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 1}},
        )
        yield _sse("content_block_start", index=0)
        yield _sse(
            "content_block_delta", index=0, delta={"type": "text_delta", "text": "Hi"}
        )
        yield _sse(
            "content_block_delta",
            index=0,
            delta={"type": "text_delta", "text": " there"},
        )
        yield _sse("content_block_stop", index=0)
        yield _sse(
            "message_delta",
            delta={"stop_reason": "end_turn"},
            usage={"output_tokens": 5},
        )
        yield _sse("message_stop")

    texts = []
    input_tokens = None
    output_tokens = None
    async for payload in parse_sse_events(fake_stream()):
        data = json.loads(payload)
        t = data.get("type")
        if t == "message_start":
            u = data["message"]["usage"]
            input_tokens = u["input_tokens"]
            output_tokens = u["output_tokens"]
        elif t == "content_block_delta":
            d = data["delta"]
            if d.get("type") == "text_delta":
                texts.append(d["text"])
        elif t == "message_delta":
            if "usage" in data:
                output_tokens = data["usage"]["output_tokens"]

    assert texts == ["Hi", " there"]
    assert input_tokens == 10
    assert output_tokens == 5
