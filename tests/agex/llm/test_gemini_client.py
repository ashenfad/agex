from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agex.agent.events import TaskStartEvent
from agex.llm.core import TokenChunk
from agex.llm.gemini_client import Gemini


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


def test_gemini_client_complete_stream():
    """Test that complete_stream properly converts events and streams tokens."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models

        # Create a mock response stream
        mock_chunk = MagicMock()
        mock_chunk.text = (
            "<TITLE>Test</TITLE><THINKING>Some thinking</THINKING><PYTHON>pass</PYTHON>"
        )
        mock_response_stream = [mock_chunk]
        mock_models.generate_content_stream.return_value = mock_response_stream

        client = Gemini()
        system = "System prompt"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        # Use complete_stream and consume the generator
        chunks = list(client.complete_stream(system, events))

        # Verify generate_content_stream was called
        mock_models.generate_content_stream.assert_called_once()
        call_kwargs = mock_models.generate_content_stream.call_args.kwargs

        # Verify pre-fill
        gemini_contents = call_kwargs["contents"]
        assert gemini_contents[-1].role == "model"
        assert "<TITLE>" in gemini_contents[-1].parts[0].text

        # Verify system format update
        config = call_kwargs["config"]
        assert "<TITLE>" in config.system_instruction

        assert len(chunks) > 0


def test_gemini_client_complete_wraps_stream():
    """Test that complete() calls complete_stream and accumulates result."""
    with patch.object(Gemini, "complete_stream") as mock_stream:
        # Mock stream tokens
        mock_stream.return_value = [
            TokenChunk(type="title", content="My Title", done=False),
            TokenChunk(type="title", content="", done=True),
            TokenChunk(type="thinking", content="Thinking...", done=False),
            TokenChunk(type="thinking", content="", done=True),
            TokenChunk(type="file", content="path=utils.py", done=False),
            TokenChunk(type="file", content="x=1", done=False),
            TokenChunk(type="file", content="", done=True),
            TokenChunk(type="python", content="pass", done=False),
            TokenChunk(type="python", content="", done=True),
        ]

        client = Gemini()
        response = client.complete("system", [])

        assert response.title == "My Title"
        assert response.thinking == "Thinking..."
        assert response.file_actions[0].path == "utils.py"
        assert response.file_actions[0].content == "x=1"
        assert response.code == "pass"


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
async def test_gemini_acomplete_stream():
    """Test async acomplete_stream method."""
    with patch("google.genai.Client") as MockClient:
        mock_aio = MockClient.return_value.aio
        mock_models = mock_aio.models

        # Create async iterator for stream
        async def mock_stream_iter():
            chunks = [
                MagicMock(text="<THINKING>"),
                MagicMock(text="Some thinking</THINKING>"),
            ]
            for chunk in chunks:
                yield chunk

        mock_models.generate_content_stream = AsyncMock(return_value=mock_stream_iter())

        client = Gemini()
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
async def test_gemini_acomplete_wraps_stream():
    """Test that acomplete() calls acomplete_stream and accumulates result."""

    # We need to mock acomplete_stream to return an async generator
    async def mock_tokens(*args, **kwargs):
        yield TokenChunk(type="thinking", content="Thinking...", done=False)
        yield TokenChunk(type="thinking", content="", done=True)
        yield TokenChunk(type="python", content="pass", done=False)
        yield TokenChunk(type="python", content="", done=True)

    with patch.object(Gemini, "acomplete_stream", side_effect=mock_tokens):
        client = Gemini()
        response = await client.acomplete("system", [])

        assert response.thinking == "Thinking..."
        assert response.code == "pass"
