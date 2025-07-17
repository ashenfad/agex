from unittest.mock import MagicMock, patch

import pytest

from agex.llm.core import LLMResponse, TextMessage
from agex.llm.gemini_client import GeminiClient


def test_gemini_client_initialization():
    """Test that GeminiClient can be initialized with default parameters."""
    client = GeminiClient()
    assert client.model == "gemini-1.5-flash"
    assert client.provider_name == "Google Gemini"
    assert client.context_window == 1048576


def test_gemini_client_custom_model():
    """Test that GeminiClient can be initialized with custom model."""
    client = GeminiClient(model="gemini-1.5-pro")
    assert client.model == "gemini-1.5-pro"
    assert client.context_window == 2097152


def test_gemini_client_message_conversion():
    """Test that messages are properly converted to Gemini format."""
    client = GeminiClient()

    # Mock the response
    mock_response = MagicMock()
    mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

        messages = [
            TextMessage(role="system", content="You are a helpful assistant."),
            TextMessage(role="user", content="Hello"),
            TextMessage(role="assistant", content="Hi there!"),
            TextMessage(role="user", content="How are you?"),
        ]

        response = client.complete(messages)  # type: ignore

        # Verify the call was made correctly
        mock_client.generate_content.assert_called_once()
        call_args = mock_client.generate_content.call_args

        # Check the converted messages
        gemini_messages = call_args[0][0]  # First positional argument

        # Should have 3 messages (system prepended to first user message)
        assert len(gemini_messages) == 3

        # First message should contain system message prepended as a separate part
        assert gemini_messages[0]["role"] == "user"
        assert (
            gemini_messages[0]["parts"][0]["text"]
            == "System: You are a helpful assistant."
        )
        assert gemini_messages[0]["parts"][1]["text"] == "Hello"

        # Second message should be assistant (mapped to "model")
        assert gemini_messages[1]["role"] == "model"
        assert gemini_messages[1]["parts"][0]["text"] == "Hi there!"

        # Third message should be user
        assert gemini_messages[2]["role"] == "user"
        assert gemini_messages[2]["parts"][0]["text"] == "How are you?"

        # Check response parsing
        assert isinstance(response, LLMResponse)
        assert response.thinking == "Test thinking"
        assert response.code == "print('hello')"


def test_gemini_client_multiple_system_messages():
    """Test that multiple system messages are properly combined."""
    client = GeminiClient()

    mock_response = MagicMock()
    mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

        messages = [
            TextMessage(role="system", content="You are a helpful assistant."),
            TextMessage(role="system", content="You are also very knowledgeable."),
            TextMessage(role="user", content="Hello"),
        ]

        response = client.complete(messages)  # type: ignore

        # Verify system messages were combined in the first user message
        call_args = mock_client.generate_content.call_args
        gemini_messages = call_args[0][0]

        first_message_parts = gemini_messages[0]["parts"]
        expected_system = (
            "You are a helpful assistant.\n\nYou are also very knowledgeable."
        )
        assert first_message_parts[0]["text"] == f"System: {expected_system}"
        assert first_message_parts[1]["text"] == "Hello"


def test_gemini_client_structured_output_config():
    """Test that structured output configuration is properly set."""
    client = GeminiClient()

    mock_response = MagicMock()
    mock_response.text = '{"thinking": "Test thinking", "code": "print(\'hello\')"}'

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

        messages = [TextMessage(role="user", content="Hello")]

        response = client.complete(messages)  # type: ignore

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

        messages = [TextMessage(role="user", content="Hello")]

        with pytest.raises(RuntimeError, match="Failed to parse Gemini JSON response"):
            client.complete(messages)  # type: ignore


def test_gemini_client_empty_response():
    """Test proper error handling for empty responses."""
    client = GeminiClient()

    mock_response = MagicMock()
    mock_response.text = ""

    with patch.object(client, "client") as mock_client:
        mock_client.generate_content.return_value = mock_response

        messages = [TextMessage(role="user", content="Hello")]

        with pytest.raises(RuntimeError, match="Gemini returned empty response"):
            client.complete(messages)  # type: ignore


def test_gemini_client_token_estimation():
    """Test token estimation method."""
    client = GeminiClient()

    text = "Hello world, this is a test message."
    estimated_tokens = client.estimate_tokens(text)

    # Should be roughly text length / 4
    expected_tokens = len(text) // 4
    assert estimated_tokens == expected_tokens


def test_gemini_client_context_window():
    """Test context window values for different models."""
    # Test default model
    client = GeminiClient()
    assert client.context_window == 1048576  # 1M tokens

    # Test Pro model
    client_pro = GeminiClient(model="gemini-1.5-pro")
    assert client_pro.context_window == 2097152  # 2M tokens

    # Test legacy model
    client_legacy = GeminiClient(model="gemini-1.0-pro")
    assert client_legacy.context_window == 32768  # 32K tokens

    # Test unknown model (should default to 1M)
    client_unknown = GeminiClient(model="gemini-unknown")
    assert client_unknown.context_window == 1048576
