from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agex.agent.events import TaskStartEvent
from agex.llm.anthropic_client import Anthropic
from agex.llm.core import TokenChunk


def test_anthropic_client_initialization():
    """Test that Anthropic client can be initialized."""
    with patch("anthropic.Anthropic"):
        client = Anthropic(api_key="test")
        assert client.model == "claude-3-sonnet-20240229"
        assert client.provider_name == "Anthropic"


def test_anthropic_client_complete_stream():
    """Test that complete_stream properly converts events and streams tokens."""
    client = Anthropic(api_key="test")

    # Mock the anthropic stream
    mock_stream = MagicMock()
    # Anthropic stream has a __enter__ that returns an object with text_stream
    # The pre-filled <TITLE> needs to be closed
    mock_stream.__enter__.return_value.text_stream = [
        "Test</TITLE><THINKING>Some thinking</THINKING>",
        "<PYTHON>pass</PYTHON>",
    ]

    with patch.object(client.client.messages, "stream", return_value=mock_stream):
        system = "You are a helpful assistant."
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        # Consume generator
        chunks = list(client.complete_stream(system, events))

        assert len(chunks) > 0
        # Should have thinking and python tokens
        assert any(c.type == "thinking" for c in chunks)
        assert any(c.type == "python" for c in chunks)


def test_anthropic_client_complete_wraps_stream():
    """Test that complete() calls complete_stream and accumulates result."""
    with patch.object(Anthropic, "complete_stream") as mock_stream:
        # Mock stream tokens
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

        assert response.title == "My Title"
        assert response.thinking == "Thinking..."
        assert response.code == "pass"


@pytest.mark.asyncio
async def test_anthropic_acomplete_stream():
    """Test async acomplete_stream method."""

    client = Anthropic(api_key="test")

    async def mock_async_iter():
        chunks = [
            MagicMock(type="content_block_delta", delta=MagicMock(text="<THINKING>")),
            MagicMock(
                type="content_block_delta", delta=MagicMock(text="Thinking</THINKING>")
            ),
        ]
        for chunk in chunks:
            yield chunk

    with patch.object(
        client.async_client.messages,
        "create",
        AsyncMock(return_value=mock_async_iter()),
    ):
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        tokens = []
        async for token in client.acomplete_stream("system", events):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(t, TokenChunk) for t in tokens)


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

        assert response.thinking == "Thinking..."
        assert response.code == "pass"
