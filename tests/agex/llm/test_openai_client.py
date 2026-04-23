from unittest.mock import patch

import pytest

from agex.llm.core import TokenChunk
from agex.llm.openai_client import OpenAI
from tests.agex._emissions import (
    response_code,
    response_thinking,
    response_title,
)


def test_openai_client_initialization():
    """Test that OpenAI client can be initialized."""
    with patch("openai.OpenAI"):
        client = OpenAI(api_key="test")
        assert client.model == "gpt-5-mini"
        assert client.provider_name == "OpenAI"


def test_reasoning_effort_default_on_tool_use_path():
    """Tool-use path should default reasoning_effort="low" when the
    caller doesn't set it — the API's own default is "none" which
    produces the wrong behaviour for agentic multi-step work."""
    from agex.llm.formats import ToolUseWireFormat

    client = OpenAI(api_key="test", wire_format=ToolUseWireFormat())

    mock_stream = []  # empty is fine; we only care about the call kwargs
    with patch.object(
        client.client.chat.completions, "create", return_value=mock_stream
    ) as create_mock:
        list(client.complete_stream("sys", []))

    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs.get("reasoning_effort") == "low"


def test_reasoning_effort_explicit_override_wins():
    """Caller-supplied ``reasoning_effort`` takes precedence over the
    default."""
    from agex.llm.formats import ToolUseWireFormat

    client = OpenAI(
        api_key="test",
        wire_format=ToolUseWireFormat(),
        reasoning_effort="high",
    )

    with patch.object(
        client.client.chat.completions, "create", return_value=[]
    ) as create_mock:
        list(client.complete_stream("sys", []))

    assert create_mock.call_args.kwargs.get("reasoning_effort") == "high"


def test_openai_client_complete_wraps_stream():
    """Test that complete() calls complete_stream and accumulates result."""
    with patch.object(OpenAI, "complete_stream") as mock_stream:
        # Mock stream tokens
        mock_stream.return_value = [
            TokenChunk(type="title", content="My Title", done=False),
            TokenChunk(type="title", content="", done=True),
            TokenChunk(type="thinking", content="Thinking...", done=False),
            TokenChunk(type="thinking", content="", done=True),
            TokenChunk(type="python", content="pass", done=False),
            TokenChunk(type="python", content="", done=True),
        ]

        client = OpenAI(api_key="test")
        response = client.complete("system", [])

        assert response_title(response) == "My Title"
        assert response_thinking(response) == "Thinking..."
        assert response_code(response) == "pass"


@pytest.mark.asyncio
async def test_openai_acomplete_wraps_stream():
    """Test that acomplete() calls acomplete_stream and accumulates result."""

    async def mock_tokens(*args, **kwargs):
        yield TokenChunk(type="thinking", content="Thinking...", done=False)
        yield TokenChunk(type="thinking", content="", done=True)
        yield TokenChunk(type="python", content="pass", done=False)
        yield TokenChunk(type="python", content="", done=True)

    with patch.object(OpenAI, "acomplete_stream", side_effect=mock_tokens):
        client = OpenAI(api_key="test")
        response = await client.acomplete("system", [])

        assert response_thinking(response) == "Thinking..."
        assert response_code(response) == "pass"
