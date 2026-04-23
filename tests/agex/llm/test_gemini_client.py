from unittest.mock import patch

import pytest

from agex.llm.core import TokenChunk
from agex.llm.gemini_client import Gemini
from tests.agex._emissions import (
    response_code,
    response_file_actions,
    response_thinking,
    response_title,
)


def test_gemini_client_initialization():
    """Test that Gemini can be initialized with default parameters."""
    with patch("google.genai.Client") as MockClient:
        client = Gemini()
        assert client.model == "gemini-1.5-flash"
        assert client.provider_name == "Google Gemini"
        MockClient.assert_called_once()


def test_gemini_client_custom_model():
    """Test that Gemini can be initialized with custom model."""
    with patch("google.genai.Client"):
        client = Gemini(model="gemini-1.5-pro")
        assert client.model == "gemini-1.5-pro"


def test_gemini_client_complete_wraps_stream():
    """Test that complete() calls complete_stream and accumulates result."""
    from agex.agent.emissions import FileWriteEmission

    with (
        patch("google.genai.Client"),
        patch.object(Gemini, "complete_stream") as mock_stream,
    ):
        # Mock stream tokens using the emission-era token shape:
        # action-tool chunks stream char-by-char, file tools arrive as
        # a single ``emission`` token carrying a prebuilt emission.
        mock_stream.return_value = [
            TokenChunk(
                type="emission",
                content="",
                done=True,
                emission_index=0,
                emission=FileWriteEmission(path="utils.py", content="x=1"),
            ),
            TokenChunk(type="title", content="My Title", done=False, emission_index=1),
            TokenChunk(type="title", content="", done=True, emission_index=1),
            TokenChunk(
                type="thinking",
                content="Thinking...",
                done=False,
                emission_index=1,
            ),
            TokenChunk(type="thinking", content="", done=True, emission_index=1),
            TokenChunk(type="python", content="pass", done=False, emission_index=1),
            TokenChunk(type="python", content="", done=True, emission_index=1),
        ]

        client = Gemini()
        response = client.complete("system", [])

        assert response_title(response) == "My Title"
        assert response_thinking(response) == "Thinking..."
        assert response_file_actions(response)[0].path == "utils.py"
        assert response_file_actions(response)[0].content == "x=1"
        assert response_code(response) == "pass"


def test_gemini_client_summarize_with_config():
    """Test that summarize handles config parameters like max_tokens correctly."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models
        mock_models.generate_content.return_value.text = "Summary text"

        client = Gemini()
        system = "Summarize this"
        content = "Some long content"

        # Call summarize with max_tokens
        client.summarize(system, content, max_tokens=100, temperature=0.5)

        # Verify generate_content was called with correct config
        mock_models.generate_content.assert_called_once()
        call_kwargs = mock_models.generate_content.call_args.kwargs

        config = call_kwargs["config"]
        assert config.max_output_tokens == 100
        assert config.temperature == 0.5


# =============================================================================
# Async Tests
# =============================================================================


@pytest.mark.asyncio
async def test_gemini_acomplete_wraps_stream():
    """Test that acomplete() calls acomplete_stream and accumulates result."""

    # We need to mock acomplete_stream to return an async generator
    async def mock_tokens(*args, **kwargs):
        yield TokenChunk(type="thinking", content="Thinking...", done=False)
        yield TokenChunk(type="thinking", content="", done=True)
        yield TokenChunk(type="python", content="pass", done=False)
        yield TokenChunk(type="python", content="", done=True)

    with (
        patch("google.genai.Client"),
        patch.object(Gemini, "acomplete_stream", side_effect=mock_tokens),
    ):
        client = Gemini()
        response = await client.acomplete("system", [])

        assert response_thinking(response) == "Thinking..."
        assert response_code(response) == "pass"
