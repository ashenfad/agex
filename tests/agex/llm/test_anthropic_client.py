from unittest.mock import patch

import pytest

from agex.llm.anthropic_client import (
    Anthropic,
    _ensure_extended_thinking,
    _ensure_tool_choice_any,
)
from agex.llm.core import TokenChunk
from tests.agex._emissions import (
    response_code,
    response_thinking,
    response_title,
)


def test_anthropic_client_initialization():
    """Test that Anthropic client can be initialized."""
    with patch("anthropic.Anthropic"):
        client = Anthropic(api_key="test")
        assert client.model == "claude-3-sonnet-20240229"
        assert client.provider_name == "Anthropic"


def test_anthropic_client_complete_wraps_stream():
    """Test that complete() calls complete_stream and accumulates result."""
    with patch.object(Anthropic, "complete_stream") as mock_stream:
        mock_stream.return_value = [
            TokenChunk(type="title", content="My Title", done=False),
            TokenChunk(type="title", content="", done=True),
            TokenChunk(type="thinking", content="Thinking...", done=False),
            TokenChunk(type="thinking", content="", done=True),
            TokenChunk(type="python", content="pass", done=False),
            TokenChunk(type="python", content="", done=True),
        ]

        client = Anthropic(api_key="test")
        response = client.complete("system", [])

        assert response_title(response) == "My Title"
        assert response_thinking(response) == "Thinking..."
        assert response_code(response) == "pass"


class TestEnsureToolChoiceAny:
    """Anthropic defaults to ``tool_choice={"type": "any"}`` so Claude
    must call one of our tools each turn — agex's loop progresses
    through tools, not prose."""

    def test_absent_choice_defaults_to_any(self):
        out = _ensure_tool_choice_any({})
        assert out["tool_choice"] == {"type": "any"}

    def test_user_supplied_choice_wins(self):
        out = _ensure_tool_choice_any({"tool_choice": {"type": "auto"}})
        assert out["tool_choice"] == {"type": "auto"}

    def test_extended_thinking_skips_force(self):
        """Claude rejects ``tool_choice != auto`` when extended thinking
        is enabled; the helper must leave the kwargs alone."""
        kwargs = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
        out = _ensure_tool_choice_any(kwargs)
        assert "tool_choice" not in out


class TestNativeThinkingDefaults:
    """Anthropic now defaults to ``native_thinking=True`` so ``thinking``
    isn't a schema parameter Claude can fill while leaving ``code``
    empty.  Enabling it requires the ``thinking`` request kwarg AND
    means ``tool_choice=any`` must be skipped (the API rejects the
    combination)."""

    def test_default_wire_format_is_native_thinking(self):
        with patch("anthropic.Anthropic"):
            client = Anthropic(api_key="test")
        assert getattr(client._wire_format, "native_thinking", False) is True

    def test_ensure_extended_thinking_defaults_the_block(self):
        out = _ensure_extended_thinking({})
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_ensure_extended_thinking_respects_user_kwarg(self):
        out = _ensure_extended_thinking({"thinking": {"type": "disabled"}})
        assert out["thinking"] == {"type": "disabled"}

    def test_native_thinking_disables_tool_choice_any_via_helper(self):
        """End-to-end: enabling thinking lands in kwargs first, so the
        subsequent ``_ensure_tool_choice_any`` call sees it and skips
        forcing (which would fail at the API)."""
        kwargs = _ensure_extended_thinking({})
        kwargs = _ensure_tool_choice_any(kwargs)
        assert "thinking" in kwargs
        assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_anthropic_acomplete_wraps_stream():
    """Test that acomplete() calls acomplete_stream and accumulates result."""

    async def mock_tokens(*args, **kwargs):
        yield TokenChunk(type="thinking", content="Thinking...", done=False)
        yield TokenChunk(type="thinking", content="", done=True)
        yield TokenChunk(type="python", content="pass", done=False)
        yield TokenChunk(type="python", content="", done=True)

    with patch.object(Anthropic, "acomplete_stream", side_effect=mock_tokens):
        client = Anthropic(api_key="test")
        response = await client.acomplete("system", [])

        assert response_thinking(response) == "Thinking..."
        assert response_code(response) == "pass"
