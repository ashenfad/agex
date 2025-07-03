from unittest.mock import MagicMock, patch

from agex.llm.anthropic_client import AnthropicClient
from agex.llm.core import LLMResponse, Message


def test_anthropic_client_initialization():
    """Test that AnthropicClient can be initialized with default parameters."""
    client = AnthropicClient()
    assert client.model == "claude-3-sonnet-20240229"
    assert client.provider_name == "Anthropic"
    assert client.context_window == 200000


def test_anthropic_client_custom_model():
    """Test that AnthropicClient can be initialized with custom model."""
    client = AnthropicClient(model="claude-3-haiku-20240307")
    assert client.model == "claude-3-haiku-20240307"


def test_anthropic_client_message_separation():
    """Test that system messages are properly separated from conversation messages."""
    client = AnthropicClient()

    # Mock the anthropic client
    mock_response = MagicMock()
    mock_tool_use = MagicMock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "structured_response"
    mock_tool_use.input = {"thinking": "Test thinking", "code": "print('hello')"}
    mock_response.content = [mock_tool_use]

    with patch.object(client, "client") as mock_client:
        mock_client.messages.create.return_value = mock_response

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]

        response = client.complete(messages)

        # Verify the call was made correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args

        # Check that system message was passed separately
        assert call_args[1]["system"] == "You are a helpful assistant."

        # Check that conversation messages were properly formatted
        conv_messages = call_args[1]["messages"]
        assert len(conv_messages) == 3  # Excluding system message
        assert conv_messages[0]["role"] == "user"
        assert conv_messages[0]["content"] == "Hello"
        assert conv_messages[1]["role"] == "assistant"
        assert conv_messages[1]["content"] == "Hi there!"
        assert conv_messages[2]["role"] == "user"
        assert conv_messages[2]["content"] == "How are you?"

        # Check that structured response tool was configured
        tools = call_args[1]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "structured_response"

        # Check response parsing
        assert isinstance(response, LLMResponse)
        assert response.thinking == "Test thinking"
        assert response.code == "print('hello')"


def test_anthropic_client_multiple_system_messages():
    """Test that multiple system messages are properly combined."""
    client = AnthropicClient()

    mock_response = MagicMock()
    mock_tool_use = MagicMock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "structured_response"
    mock_tool_use.input = {"thinking": "Test thinking", "code": "print('hello')"}
    mock_response.content = [mock_tool_use]

    with patch.object(client, "client") as mock_client:
        mock_client.messages.create.return_value = mock_response

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="system", content="You are also very knowledgeable."),
            Message(role="user", content="Hello"),
        ]

        response = client.complete(messages)

        # Verify system messages were combined
        call_args = mock_client.messages.create.call_args
        expected_system = (
            "You are a helpful assistant.\n\nYou are also very knowledgeable."
        )
        assert call_args[1]["system"] == expected_system


def test_anthropic_client_token_estimation():
    """Test token estimation method."""
    client = AnthropicClient()

    text = "Hello world, this is a test message."
    estimated_tokens = client.estimate_tokens(text)

    # Should be roughly text length / 4
    expected_tokens = len(text) // 4
    assert estimated_tokens == expected_tokens
