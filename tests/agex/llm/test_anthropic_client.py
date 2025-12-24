from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agex.agent.events import ActionEvent, TaskStartEvent
from agex.llm.anthropic_client import Anthropic
from agex.llm.core import LLMResponse, TokenChunk


def test_anthropic_client_initialization():
    """Test that Anthropic can be initialized with default parameters."""
    client = Anthropic()
    assert client.model == "claude-3-sonnet-20240229"
    assert client.provider_name == "Anthropic"


def test_anthropic_client_custom_model():
    """Test that Anthropic can be initialized with custom model."""
    client = Anthropic(model="claude-3-haiku-20240307")
    assert client.model == "claude-3-haiku-20240307"


def test_anthropic_client_event_handling():
    """Test that events are properly handled by Anthropic API."""
    client = Anthropic()

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
    client = Anthropic()

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


# =============================================================================
# Async Tests
# =============================================================================


@pytest.mark.asyncio
async def test_anthropic_acomplete():
    """Test async acomplete method."""
    client = Anthropic()

    mock_response = MagicMock()
    mock_tool_use = MagicMock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "structured_response"
    mock_tool_use.input = {"thinking": "Async thinking", "code": "print('async')"}
    mock_response.content = [mock_tool_use]

    with patch.object(client, "async_client") as mock_async:
        mock_async.messages.create = AsyncMock(return_value=mock_response)

        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        result = await client.acomplete("system", events)

        assert isinstance(result, LLMResponse)
        assert result.thinking == "Async thinking"
        assert result.code == "print('async')"
        mock_async.messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_anthropic_acomplete_empty_response():
    """Test async acomplete error for empty response."""
    client = Anthropic()

    mock_response = MagicMock()
    mock_response.content = []

    with patch.object(client, "async_client") as mock_async:
        mock_async.messages.create = AsyncMock(return_value=mock_response)

        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        with pytest.raises(RuntimeError, match="Anthropic returned empty response"):
            await client.acomplete("system", events)


@pytest.mark.asyncio
async def test_anthropic_acomplete_api_error():
    """Test async acomplete error handling."""
    client = Anthropic()

    with patch.object(client, "async_client") as mock_async:
        mock_async.messages.create = AsyncMock(side_effect=Exception("API Error"))

        events = [
            TaskStartEvent(
                agent_name="test", task_name="test", inputs={}, message="Hello"
            )
        ]

        with pytest.raises(RuntimeError, match="Anthropic completion failed"):
            await client.acomplete("system", events)


@pytest.mark.asyncio
async def test_anthropic_acomplete_stream():
    """Test async acomplete_stream method."""
    client = Anthropic()

    # Mock streaming events
    async def mock_stream():
        events = [
            MagicMock(type="content_block_delta", delta=MagicMock(text="<THINKING>")),
            MagicMock(
                type="content_block_delta", delta=MagicMock(text="Some thinking")
            ),
            MagicMock(type="content_block_delta", delta=MagicMock(text="</THINKING>")),
        ]
        for event in events:
            yield event

    with patch.object(client, "async_client") as mock_async:
        mock_async.messages.create = AsyncMock(return_value=mock_stream())

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
