from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from agex.agent.events import ActionEvent, TaskStartEvent
from agex.llm.core import LLMResponse, TokenChunk
from agex.llm.gemini_client import GeminiClient


def test_gemini_client_initialization():
    """Test that GeminiClient can be initialized with default parameters."""
    with patch("google.genai.Client") as MockClient:
        client = GeminiClient()
        assert client.model == "gemini-1.5-flash"
        assert client.provider_name == "Google Gemini"
        MockClient.assert_called_once()


def test_gemini_client_custom_model():
    """Test that GeminiClient can be initialized with custom model."""
    with patch("google.genai.Client"):
        client = GeminiClient(model="gemini-1.5-pro")
        assert client.model == "gemini-1.5-pro"


def test_gemini_client_event_handling():
    """Test that events are properly converted to Gemini format."""
    with patch("google.genai.Client") as MockClient:
        # Mock instance and models service
        mock_client_instance = MockClient.return_value
        mock_models = mock_client_instance.models

        # Mock the response
        mock_response = MagicMock()
        mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'
        mock_models.generate_content.return_value = mock_response

        client = GeminiClient()

        system = "You are a helpful assistant."
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            ),
            ActionEvent(
                agent_name="test", thinking="Hi there!", code="task_continue()"
            ),
        ]

        response = client.complete(system, events)

        # Verify the call was made correctly
        mock_models.generate_content.assert_called_once()
        call_kwargs = mock_models.generate_content.call_args.kwargs

        # Check model and contents
        assert call_kwargs["model"] == "gemini-1.5-flash"
        gemini_contents = call_kwargs["contents"]
        assert len(gemini_contents) >= 1
        assert isinstance(gemini_contents[0], types.Content)
        assert gemini_contents[0].role == "user"

        # Check config
        config = call_kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.system_instruction == system
        assert config.response_mime_type == "application/json"

        # Check response parsing
        assert isinstance(response, LLMResponse)
        assert response.thinking == "Test thinking"
        assert response.code == "print('hello')"


def test_gemini_client_system_message():
    """Test that system message is passed in config."""
    with patch("google.genai.Client") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_models = mock_client_instance.models

        mock_response = MagicMock()
        mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'
        mock_models.generate_content.return_value = mock_response

        client = GeminiClient()

        system = "System Instructions"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        client.complete(system, events)

        # Verify system passed in config
        call_kwargs = mock_models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.system_instruction == system


def test_gemini_client_structured_output_config():
    """Test that structured output configuration is properly set."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models

        mock_response = MagicMock()
        mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'
        mock_models.generate_content.return_value = mock_response

        client = GeminiClient()
        system = "Test"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        client.complete(system, events)

        # Verify config
        call_kwargs = mock_models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_schema is not None
        assert config.response_schema["required"] == ["thinking", "code"]


def test_gemini_client_json_parsing_error():
    """Test proper error handling for invalid JSON responses."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models
        mock_models.generate_content.return_value.text = "invalid json"

        client = GeminiClient()
        system = "Test"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        with pytest.raises(RuntimeError, match="Failed to parse Gemini JSON response"):
            client.complete(system, events)


def test_gemini_client_empty_response():
    """Test proper error handling for empty responses."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models
        mock_models.generate_content.return_value.text = ""

        client = GeminiClient()
        system = "Test"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        with pytest.raises(RuntimeError, match="Gemini returned empty response"):
            client.complete(system, events)


def test_gemini_client_complete_stream():
    """Test that complete_stream properly converts events and streams tokens."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models

        # Create a mock response stream
        mock_chunk = MagicMock()
        mock_chunk.text = "<THINKING>Some thinking</THINKING>"
        mock_response_stream = [mock_chunk]
        mock_models.generate_content_stream.return_value = mock_response_stream

        client = GeminiClient()
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


def test_gemini_client_summarize_with_config():
    """Test that summarize handles config parameters like max_tokens correctly."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models
        mock_models.generate_content.return_value.text = "Summary text"

        client = GeminiClient()
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


def test_gemini_client_google_search_enabled():
    """Test that google_search parameter enables the Google Search tool."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models
        mock_response = MagicMock()
        mock_response.text = '{"thinking": "Test", "code": "pass"}'
        mock_models.generate_content.return_value = mock_response

        # Initialize with google_search=True
        client = GeminiClient(google_search=True)

        system = "Test System"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        client.complete(system, events)

        # Verify generate_content was called
        mock_models.generate_content.assert_called_once()
        call_kwargs = mock_models.generate_content.call_args.kwargs

        # Verify config has tools
        config = call_kwargs["config"]
        assert config.tools is not None
        assert len(config.tools) == 1

        # Verify system prompt has grounding primer
        assert "# Grounding Tools Enabled" in config.system_instruction
        assert "- Google Search" in config.system_instruction


def test_gemini_client_google_search_stream():
    """Test that google_search parameter affects streaming."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models

        # Mock stream response
        mock_chunk = MagicMock()
        mock_chunk.text = "<THINKING>Thinking</THINKING>"
        mock_models.generate_content_stream.return_value = [mock_chunk]

        client = GeminiClient(google_search=True)
        system = "Test"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        # Consume stream
        list(client.complete_stream(system, events))

        # Verify config
        call_kwargs = mock_models.generate_content_stream.call_args.kwargs
        config = call_kwargs["config"]

        assert config.tools is not None
        assert "# Grounding Tools Enabled" in config.system_instruction
        assert "- Google Search" in config.system_instruction


def test_gemini_client_url_context():
    """Test that url_context parameter passes the correct tool config."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models
        mock_response = MagicMock()
        mock_response.text = '{"thinking": "Test", "code": "pass"}'
        mock_models.generate_content.return_value = mock_response

        # Initialize with url_context=True
        client = GeminiClient(url_context=True)

        system = "Test System"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        client.complete(system, events)

        # Verify generate_content was called
        mock_models.generate_content.assert_called_once()
        call_kwargs = mock_models.generate_content.call_args.kwargs

        # Verify config has tools
        config = call_kwargs["config"]
        assert config.tools is not None
        # Check that one of the tools has url_context configured
        # The SDK converts the dict to a Tool object with a url_context attribute
        assert any(
            getattr(tool, "url_context", None) is not None for tool in config.tools
        )

        # Verify system prompt has grounding primer
        assert "# Grounding Tools Enabled" in config.system_instruction
        assert "- URL Context" in config.system_instruction


def test_gemini_client_url_context_stream_prefill():
    """Test that url_context disables pre-fill in streaming."""
    with patch("google.genai.Client") as MockClient:
        mock_models = MockClient.return_value.models

        # Mock stream response
        mock_chunk = MagicMock()
        mock_chunk.text = "<THINKING>Thinking</THINKING>"
        mock_models.generate_content_stream.return_value = [mock_chunk]

        client = GeminiClient(url_context=True)
        system = "Test"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        # Consume stream
        list(client.complete_stream(system, events))

        # Verify generate_content_stream call arguments
        call_kwargs = mock_models.generate_content_stream.call_args.kwargs
        gemini_contents = call_kwargs["contents"]

        # Check that pre-fill content was NOT added to the request
        # The last message should be the user message, not a model pre-fill
        assert gemini_contents[-1].role == "user"


# =============================================================================
# Async Tests
# =============================================================================


@pytest.mark.asyncio
async def test_gemini_acomplete():
    """Test async acomplete method."""
    with patch("google.genai.Client") as MockClient:
        mock_aio = MockClient.return_value.aio
        mock_models = mock_aio.models

        mock_response = MagicMock()
        mock_response.text = (
            '{"thinking": "Async thinking", "code": "print(\'async\')"}'
        )
        mock_models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiClient()
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        result = await client.acomplete("system", events)

        assert isinstance(result, LLMResponse)
        assert result.thinking == "Async thinking"
        assert result.code == "print('async')"
        mock_models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_acomplete_empty_response():
    """Test async acomplete error for empty response."""
    with patch("google.genai.Client") as MockClient:
        mock_aio = MockClient.return_value.aio
        mock_models = mock_aio.models

        mock_response = MagicMock()
        mock_response.text = ""
        mock_models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiClient()
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        with pytest.raises(RuntimeError, match="Gemini returned empty response"):
            await client.acomplete("system", events)


@pytest.mark.asyncio
async def test_gemini_acomplete_stream():
    """Test async acomplete_stream method."""
    with patch("google.genai.Client") as MockClient:
        mock_aio = MockClient.return_value.aio
        mock_models = mock_aio.models

        # Create async iterator for stream
        async def mock_stream():
            chunks = [
                MagicMock(text="<THINKING>"),
                MagicMock(text="Some thinking</THINKING>"),
            ]
            for chunk in chunks:
                yield chunk

        mock_models.generate_content_stream = AsyncMock(return_value=mock_stream())

        client = GeminiClient()
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
        mock_models.generate_content_stream.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_acomplete_api_error():
    """Test async acomplete error handling."""
    with patch("google.genai.Client") as MockClient:
        mock_aio = MockClient.return_value.aio
        mock_models = mock_aio.models
        mock_models.generate_content = AsyncMock(side_effect=Exception("API Error"))

        client = GeminiClient()
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        with pytest.raises(RuntimeError, match="Gemini completion failed"):
            await client.acomplete("system", events)
