from unittest.mock import MagicMock, patch

import pytest

from agex.llm.core import LLMResponse, TextMessage
from agex.llm.openai_client import OpenAIClient


def test_openai_client_initialization():
    """Test that OpenAIClient can be initialized with default parameters."""
    client = OpenAIClient(api_key="test")
    assert client.model == "gpt-4.1-nano"
    assert client.provider_name == "OpenAI"


def test_openai_client_custom_model():
    """Test that OpenAIClient can be initialized with custom model."""
    client = OpenAIClient(model="gpt-4.1", api_key="test")
    assert client.model == "gpt-4.1"


def test_openai_client_message_handling():
    """Test that messages are properly handled by OpenAI API."""
    client = OpenAIClient(api_key="test")

    # Mock the OpenAI response
    mock_response = MagicMock()
    mock_parsed_response = LLMResponse(thinking="Test thinking", code="print('hello')")
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = mock_parsed_response

    with patch.object(client, "client") as mock_client:
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [
            TextMessage(role="system", content="You are a helpful assistant."),
            TextMessage(role="user", content="Hello"),
            TextMessage(role="assistant", content="Hi there!"),
            TextMessage(role="user", content="How are you?"),
        ]

        response = client.complete(messages)  # type: ignore

        # Verify the call was made correctly
        mock_client.beta.chat.completions.parse.assert_called_once()
        call_args = mock_client.beta.chat.completions.parse.call_args

        # Check that all messages were passed correctly
        passed_messages = call_args[1]["messages"]
        assert len(passed_messages) == 4
        assert passed_messages[0]["role"] == "system"
        assert passed_messages[0]["content"] == "You are a helpful assistant."
        assert passed_messages[1]["role"] == "user"
        assert passed_messages[1]["content"] == "Hello"
        assert passed_messages[2]["role"] == "assistant"
        assert passed_messages[2]["content"] == "Hi there!"
        assert passed_messages[3]["role"] == "user"
        assert passed_messages[3]["content"] == "How are you?"

        # Check that structured output was configured
        assert call_args[1]["response_format"] == LLMResponse
        assert call_args[1]["model"] == "gpt-4.1-nano"

        # Check response parsing
        assert isinstance(response, LLMResponse)
        assert response.thinking == "Test thinking"
        assert response.code == "print('hello')"


def test_openai_client_structured_output():
    """Test that structured output configuration is properly set."""
    client = OpenAIClient(api_key="test")

    mock_response = MagicMock()
    mock_parsed_response = LLMResponse(thinking="Test thinking", code="print('hello')")
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = mock_parsed_response

    with patch.object(client, "client") as mock_client:
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [TextMessage(role="user", content="Hello")]

        response = client.complete(messages)  # type: ignore

        # Verify structured output configuration
        call_args = mock_client.beta.chat.completions.parse.call_args
        assert call_args[1]["response_format"] == LLMResponse

        # Check response
        assert isinstance(response, LLMResponse)
        assert response.thinking == "Test thinking"
        assert response.code == "print('hello')"


def test_openai_client_request_parameters():
    """Test that additional request parameters are properly passed."""
    client = OpenAIClient(temperature=0.5, max_tokens=1000, api_key="test")

    mock_response = MagicMock()
    mock_parsed_response = LLMResponse(thinking="Test thinking", code="print('hello')")
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = mock_parsed_response

    with patch.object(client, "client") as mock_client:
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [TextMessage(role="user", content="Hello")]

        # Call with additional parameters
        response = client.complete(messages, top_p=0.9)  # type: ignore

        # Verify parameters were passed
        call_args = mock_client.beta.chat.completions.parse.call_args
        assert call_args[1]["temperature"] == 0.5  # From constructor
        assert call_args[1]["max_tokens"] == 1000  # From constructor
        assert call_args[1]["top_p"] == 0.9  # From method call


def test_openai_client_none_parsed_response():
    """Test proper error handling when OpenAI returns None for parsed response."""
    client = OpenAIClient(api_key="test")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = None

    with patch.object(client, "client") as mock_client:
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [TextMessage(role="user", content="Hello")]

        with pytest.raises(
            RuntimeError, match="OpenAI returned None for parsed response"
        ):
            client.complete(messages)  # type: ignore


def test_openai_client_api_error():
    """Test proper error handling for OpenAI API errors."""
    client = OpenAIClient(api_key="test")

    with patch.object(client, "client") as mock_client:
        mock_client.beta.chat.completions.parse.side_effect = Exception("API Error")

        messages = [TextMessage(role="user", content="Hello")]

        with pytest.raises(RuntimeError, match="OpenAI completion failed: API Error"):
            client.complete(messages)  # type: ignore


def test_openai_client_kwargs_filtering():
    """Test that provider-specific kwargs are properly filtered."""
    client = OpenAIClient(
        provider="openai", model="gpt-4.1", temperature=0.3, api_key="test"
    )

    # provider should be filtered out, others should remain
    assert client._model == "gpt-4.1"
    assert client._kwargs["temperature"] == 0.3
    assert "provider" not in client._kwargs


def test_openai_client_message_dataclass_conversion():
    """Test that Message dataclasses are properly converted to dicts."""
    client = OpenAIClient(api_key="test")

    mock_response = MagicMock()
    mock_parsed_response = LLMResponse(thinking="Test thinking", code="print('hello')")
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = mock_parsed_response

    with patch.object(client, "client") as mock_client:
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [
            TextMessage(role="system", content="System message"),
            TextMessage(role="user", content="User message"),
        ]

        client.complete(messages)  # type: ignore

        # Verify messages were converted to dicts
        call_args = mock_client.beta.chat.completions.parse.call_args
        passed_messages = call_args[1]["messages"]

        # Should be a list of dicts, not Message objects
        assert isinstance(passed_messages, list)
        assert all(isinstance(msg, dict) for msg in passed_messages)
        assert passed_messages[0] == {"role": "system", "content": "System message"}
        assert passed_messages[1] == {"role": "user", "content": "User message"}
