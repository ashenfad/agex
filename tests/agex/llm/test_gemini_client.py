from unittest.mock import MagicMock, patch

import pytest

from agex.agent.events import ActionEvent, TaskStartEvent
from agex.llm.core import LLMResponse
from agex.llm.gemini_client import GeminiClient


def test_gemini_client_initialization():
    """Test that GeminiClient can be initialized with default parameters."""
    client = GeminiClient()
    assert client.model == "gemini-1.5-flash"
    assert client.provider_name == "Google Gemini"


def test_gemini_client_custom_model():
    """Test that GeminiClient can be initialized with custom model."""
    client = GeminiClient(model="gemini-1.5-pro")
    assert client.model == "gemini-1.5-pro"


def test_gemini_client_event_handling():
    """Test that events are properly converted to Gemini format."""
    client = GeminiClient()

    # Mock the response
    mock_response = MagicMock()
    mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

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
        mock_client.generate_content.assert_called_once()
        call_args = mock_client.generate_content.call_args

        # Check the converted messages
        gemini_messages = call_args[0][0]  # First positional argument

        # Should have messages with system prepended to first user message
        assert len(gemini_messages) >= 1

        # First message should contain system message prepended as a separate part
        assert gemini_messages[0]["role"] == "user"
        assert "System:" in gemini_messages[0]["parts"][0]["text"]

        # Check response parsing
        assert isinstance(response, LLMResponse)
        assert response.thinking == "Test thinking"
        assert response.code == "print('hello')"


def test_gemini_client_system_message():
    """Test that system message is properly prepended to first user message."""
    client = GeminiClient()

    mock_response = MagicMock()
    mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

        system = "You are a helpful assistant. You are also very knowledgeable."
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        response = client.complete(system, events)

        # Verify we got a response
        assert response is not None
        assert isinstance(response, LLMResponse)

        # Verify system message was prepended in the first user message
        call_args = mock_client.generate_content.call_args
        gemini_messages = call_args[0][0]

        first_message_parts = gemini_messages[0]["parts"]
        assert first_message_parts[0]["text"] == f"System: {system}"


def test_gemini_client_structured_output_config():
    """Test that structured output configuration is properly set."""
    client = GeminiClient()

    mock_response = MagicMock()
    mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

        system = "Test"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        response = client.complete(system, events)

        # Verify we got a response
        assert response is not None
        assert isinstance(response, LLMResponse)

        # Verify generation config was set correctly
        call_args = mock_client.generate_content.call_args
        generation_config = call_args[1]["generation_config"]

        # Check structured output settings
        assert generation_config.response_mime_type == "application/json"
        assert generation_config.response_schema is not None

        # Check schema structure
        schema = generation_config.response_schema
        assert schema["type"] == "object"
        assert "thinking" in schema["properties"]
        assert "code" in schema["properties"]
        assert schema["required"] == ["thinking", "code"]


def test_gemini_client_json_parsing_error():
    """Test proper error handling for invalid JSON responses."""
    client = GeminiClient()

    mock_response = MagicMock()
    mock_response.text = "invalid json"

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

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
    client = GeminiClient()

    mock_response = MagicMock()
    mock_response.text = ""

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

        system = "Test"
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        with pytest.raises(RuntimeError, match="Gemini returned empty response"):
            client.complete(system, events)
