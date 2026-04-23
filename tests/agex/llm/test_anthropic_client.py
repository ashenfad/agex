from unittest.mock import patch

import pytest

from agex.llm.anthropic_client import Anthropic
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
