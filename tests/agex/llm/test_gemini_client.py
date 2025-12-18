from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from agex.agent.events import ActionEvent, TaskStartEvent
from agex.llm.core import LLMResponse
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
