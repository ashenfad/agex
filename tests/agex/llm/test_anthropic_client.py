from unittest.mock import MagicMock, patch

from agex.agent.events import ActionEvent, TaskStartEvent
from agex.llm.anthropic_client import AnthropicClient
from agex.llm.core import LLMResponse


def test_anthropic_client_initialization():
    """Test that AnthropicClient can be initialized with default parameters."""
    client = AnthropicClient()
    assert client.model == "claude-3-sonnet-20240229"
    assert client.provider_name == "Anthropic"


def test_anthropic_client_custom_model():
    """Test that AnthropicClient can be initialized with custom model."""
    client = AnthropicClient(model="claude-3-haiku-20240307")
    assert client.model == "claude-3-haiku-20240307"


def test_anthropic_client_event_handling():
    """Test that events are properly handled by Anthropic API."""
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

        system = "You are a helpful assistant."
        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            ),
            ActionEvent(agent_name="test", thinking="Thinking...", code="result = 1"),
        ]

        response = client.complete(system, events)

        # Verify the call was made correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args

        # Check that system message was passed separately
        assert call_args[1]["system"] == system

        # Check that messages were properly formatted (Anthropic expects content blocks)
        conv_messages = call_args[1]["messages"]
        assert len(conv_messages) >= 1

        # Check that structured response tool was configured
        tools = call_args[1]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "structured_response"

        # Check response parsing
        assert isinstance(response, LLMResponse)
        assert response.thinking == "Test thinking"
        assert response.code == "print('hello')"


def test_anthropic_client_system_message():
    """Test that system message is properly passed to Anthropic API."""
    client = AnthropicClient()

    mock_response = MagicMock()
    mock_tool_use = MagicMock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "structured_response"
    mock_tool_use.input = {"thinking": "Test thinking", "code": "print('hello')"}
    mock_response.content = [mock_tool_use]

    with patch.object(client, "client") as mock_client:
        mock_client.messages.create.return_value = mock_response

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

        # Verify system message was passed
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["system"] == system
