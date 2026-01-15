from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agex.agent.events import TaskStartEvent
from agex.llm.core import TokenChunk
from agex.llm.openai_client import OpenAI


def test_openai_client_initialization():
    """Test that OpenAI client can be initialized."""
    with patch("openai.OpenAI"):
        client = OpenAI(api_key="test")
        assert client.model == "gpt-4.1-nano"
        assert client.provider_name == "OpenAI"


def test_openai_client_complete_stream():
    """Test that complete_stream properly converts events and streams tokens."""
    client = OpenAI(api_key="test")

    # Mock the openai stream
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[
        0
    ].delta.content = "<THINKING>Some thinking</THINKING><PYTHON>pass</PYTHON>"
    mock_stream = [mock_chunk]

    with patch.object(
        client.client.chat.completions, "create", return_value=mock_stream
    ):
        system = "You are a helpful assistant."
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        # Consume generator
        chunks = list(client.complete_stream(system, events))

        assert len(chunks) > 0
        assert any(c.type == "thinking" for c in chunks)
        assert any(c.type == "python" for c in chunks)


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

        assert response.title == "My Title"
        assert response.thinking == "Thinking..."
        assert response.code == "pass"


@pytest.mark.asyncio
async def test_openai_acomplete_stream():
    """Test async acomplete_stream method."""
    client = OpenAI(api_key="test")

    # Mock the async stream
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "<THINKING>Thinking</THINKING>"

    async def mock_async_iter():
        yield mock_chunk

    with patch.object(
        client.async_client.chat.completions,
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

        assert response.thinking == "Thinking..."
        assert response.code == "pass"
