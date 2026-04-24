from unittest.mock import patch

import pytest

from agex.llm.core import TokenChunk
from agex.llm.openai_client import OpenAI
from tests.agex._emissions import (
    response_code,
    response_thinking,
    response_title,
)


def test_openai_client_initialization():
    """Test that OpenAI client can be initialized."""
    with patch("openai.OpenAI"):
        client = OpenAI(api_key="test")
        assert client.model == "gpt-5-mini"
        assert client.provider_name == "OpenAI"


def test_chat_path_does_not_inject_reasoning_effort():
    """The Chat Completions path only runs for non-reasoning models
    (gpt-5* / o1* / o3* dispatch to Responses), and those models
    reject ``reasoning_effort`` with a 400.  Verify we don't inject
    one by default."""
    from agex.llm.formats import ToolUseWireFormat

    client = OpenAI(
        api_key="test",
        wire_format=ToolUseWireFormat(),
        use_responses=False,
    )

    mock_stream = []  # empty is fine; we only care about the call kwargs
    with patch.object(
        client.client.chat.completions, "create", return_value=mock_stream
    ) as create_mock:
        list(client.complete_stream("sys", []))

    call_kwargs = create_mock.call_args.kwargs
    assert "reasoning_effort" not in call_kwargs


def test_reasoning_effort_explicit_opt_in_chat_path():
    """Callers who've forced a reasoning model onto Chat Completions
    (``use_responses=False``) can still pass ``reasoning_effort``
    explicitly — that we forward as-is."""
    from agex.llm.formats import ToolUseWireFormat

    client = OpenAI(
        api_key="test",
        wire_format=ToolUseWireFormat(),
        use_responses=False,
        reasoning_effort="high",
    )

    with patch.object(
        client.client.chat.completions, "create", return_value=[]
    ) as create_mock:
        list(client.complete_stream("sys", []))

    assert create_mock.call_args.kwargs.get("reasoning_effort") == "high"


def test_reasoning_default_on_responses_path():
    """Responses path defaults the nested ``reasoning={"effort": "medium"}``
    block and enables encrypted-content round-trip."""
    from agex.llm.formats import ToolUseWireFormat

    client = OpenAI(
        api_key="test",
        model="gpt-5-mini",
        wire_format=ToolUseWireFormat(),
    )
    assert client._use_responses is True

    with patch.object(
        client.client.responses, "create", return_value=[]
    ) as create_mock:
        list(client.complete_stream("sys", []))

    kwargs = create_mock.call_args.kwargs
    assert kwargs.get("reasoning") == {"effort": "medium"}
    assert kwargs.get("store") is False
    assert "reasoning.encrypted_content" in kwargs.get("include", [])
    assert kwargs.get("tool_choice") == "required"
    # Flat Responses tool shape — no ``function`` wrapper.
    tools = kwargs.get("tools")
    assert tools and all(t.get("type") == "function" and "name" in t for t in tools)


def test_reasoning_effort_legacy_kwarg_folds_into_responses_block():
    """Users whose code targeted Chat Completions often pass
    ``reasoning_effort="medium"`` directly.  On the Responses path we
    fold that into the nested ``reasoning`` shape so nothing breaks."""
    from agex.llm.formats import ToolUseWireFormat

    client = OpenAI(
        api_key="test",
        model="gpt-5-mini",
        wire_format=ToolUseWireFormat(),
        reasoning_effort="medium",
    )

    with patch.object(
        client.client.responses, "create", return_value=[]
    ) as create_mock:
        list(client.complete_stream("sys", []))

    kwargs = create_mock.call_args.kwargs
    assert kwargs.get("reasoning") == {"effort": "medium"}
    # The legacy kwarg shouldn't leak through alongside the nested shape.
    assert "reasoning_effort" not in kwargs


def test_use_responses_override_forces_chat():
    """``use_responses=False`` stays on Chat Completions even for
    gpt-5 models."""
    from agex.llm.formats import ToolUseWireFormat

    client = OpenAI(
        api_key="test",
        model="gpt-5-mini",
        wire_format=ToolUseWireFormat(),
        use_responses=False,
    )
    assert client._use_responses is False

    with patch.object(
        client.client.chat.completions, "create", return_value=[]
    ) as create_mock:
        list(client.complete_stream("sys", []))

    assert create_mock.called


def test_non_reasoning_model_routes_to_chat():
    """Auto-detection: a gpt-4 model uses Chat Completions."""
    client = OpenAI(api_key="test", model="gpt-4o-mini")
    assert client._use_responses is False


def test_openai_client_complete_wraps_stream():
    """Test that complete() calls complete_stream and accumulates result."""
    with patch.object(OpenAI, "complete_stream") as mock_stream:
        # Mock stream tokens
        mock_stream.return_value = [
            TokenChunk(type="title", content="My Title", done=False),
            TokenChunk(type="title", content="", done=True),
            TokenChunk(type="thinking", content="Thinking...", done=False),
            TokenChunk(type="thinking", content="", done=True),
            TokenChunk(type="python", content="pass", done=False),
            TokenChunk(type="python", content="", done=True),
        ]

        client = OpenAI(api_key="test")
        response = client.complete("system", [])

        assert response_title(response) == "My Title"
        assert response_thinking(response) == "Thinking..."
        assert response_code(response) == "pass"


@pytest.mark.asyncio
async def test_openai_acomplete_wraps_stream():
    """Test that acomplete() calls acomplete_stream and accumulates result."""

    async def mock_tokens(*args, **kwargs):
        yield TokenChunk(type="thinking", content="Thinking...", done=False)
        yield TokenChunk(type="thinking", content="", done=True)
        yield TokenChunk(type="python", content="pass", done=False)
        yield TokenChunk(type="python", content="", done=True)

    with patch.object(OpenAI, "acomplete_stream", side_effect=mock_tokens):
        client = OpenAI(api_key="test")
        response = await client.acomplete("system", [])

        assert response_thinking(response) == "Thinking..."
        assert response_code(response) == "pass"
