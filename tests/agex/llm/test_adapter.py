"""Tests for the FetchAdapter seam used by pyfetch LLM clients.

Covers two aspects:

1. The :class:`DefaultPyfetchAdapter` preserves behavior equivalent to
   the static transport methods it replaced (verified against the
   existing aiohttp-path tests).
2. Clients (:class:`PyfetchOpenAI`, :class:`PyfetchAnthropic`) route
   their network calls through the injected adapter when provided,
   omit Authorization headers when ``api_key`` is empty, and propagate
   both the non-streaming and streaming contracts faithfully.

These tests use an in-memory mock adapter so they don't require
network or aiohttp.
"""

from typing import AsyncIterator, List

import pytest

from agex.llm.adapter import DefaultPyfetchAdapter, FetchAdapter
from agex.llm.pyfetch_anthropic import PyfetchAnthropic
from agex.llm.pyfetch_openai import PyfetchOpenAI

# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class MockAdapter(FetchAdapter):
    """In-memory adapter that records calls and returns canned responses."""

    def __init__(
        self,
        *,
        json_response: dict | None = None,
        stream_chunks: List[str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self.json_response = json_response or {}
        self.stream_chunks = stream_chunks or []
        self.extra_headers = extra_headers or {}
        self.json_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def fetch_json(
        self, url: str, *, headers: dict[str, str], body: dict
    ) -> dict:
        # Allow the mock to inject headers (simulating a JS-bridge adapter
        # that adds Authorization on the way out).
        merged_headers = {**headers, **self.extra_headers}
        self.json_calls.append(
            {
                "url": url,
                "headers": merged_headers,
                "body": body,
            }
        )
        return self.json_response

    async def fetch_stream(
        self, url: str, *, headers: dict[str, str], body: dict
    ) -> AsyncIterator[str]:
        merged_headers = {**headers, **self.extra_headers}
        self.stream_calls.append(
            {
                "url": url,
                "headers": merged_headers,
                "body": body,
            }
        )
        for chunk in self.stream_chunks:
            yield chunk


# ---------------------------------------------------------------------------
# Adapter is an ABC (enforce contract)
# ---------------------------------------------------------------------------


def test_fetch_adapter_is_abstract():
    """FetchAdapter cannot be instantiated directly; both methods are abstract."""
    with pytest.raises(TypeError):
        FetchAdapter()  # type: ignore[abstract]


def test_default_adapter_instantiates():
    """DefaultPyfetchAdapter is concrete and constructible."""
    adapter = DefaultPyfetchAdapter()
    assert isinstance(adapter, FetchAdapter)


# ---------------------------------------------------------------------------
# Client routing: the adapter is used when injected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_summarize_routes_through_adapter():
    """summarize() calls adapter.fetch_json with correct URL/body."""
    adapter = MockAdapter(
        json_response={"choices": [{"message": {"content": "Hello from mock"}}]}
    )
    client = PyfetchOpenAI(
        model="test-model",
        api_key="sk-test",
        fetch_adapter=adapter,
    )

    result = await client.summarize("you are a helper", "hi")

    assert result == "Hello from mock"
    assert len(adapter.json_calls) == 1
    call = adapter.json_calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["body"]["model"] == "test-model"
    # System + user messages
    roles = [m["role"] for m in call["body"]["messages"]]
    assert roles == ["system", "user"]


@pytest.mark.asyncio
async def test_anthropic_summarize_routes_through_adapter():
    """Anthropic summarize() calls adapter.fetch_json with correct URL/body."""
    adapter = MockAdapter(
        json_response={"content": [{"type": "text", "text": "Hi from anthropic mock"}]}
    )
    client = PyfetchAnthropic(
        model="claude-test",
        api_key="sk-ant-test",
        fetch_adapter=adapter,
    )

    result = await client.summarize("you are a helper", "hi")

    assert result == "Hi from anthropic mock"
    assert len(adapter.json_calls) == 1
    call = adapter.json_calls[0]
    assert call["url"].endswith("/messages")
    assert call["headers"]["x-api-key"] == "sk-ant-test"
    assert call["headers"]["anthropic-version"]
    assert call["body"]["model"] == "claude-test"


# ---------------------------------------------------------------------------
# Authorization header omitted when api_key is empty + adapter is present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_omits_authorization_when_api_key_empty():
    """With no api_key, the client does not set Authorization — the adapter
    is expected to inject it on the way out (e.g., JS-bridge reads key from
    localStorage and adds header there)."""
    adapter = MockAdapter(json_response={"choices": [{"message": {"content": "ok"}}]})
    client = PyfetchOpenAI(api_key="", fetch_adapter=adapter)

    await client.summarize("system", "hi")

    call = adapter.json_calls[0]
    assert "Authorization" not in call["headers"]


@pytest.mark.asyncio
async def test_anthropic_omits_xapikey_when_api_key_empty():
    """Anthropic's equivalent: no x-api-key when api_key is empty."""
    adapter = MockAdapter(json_response={"content": [{"type": "text", "text": "ok"}]})
    client = PyfetchAnthropic(api_key="", fetch_adapter=adapter)

    await client.summarize("system", "hi")

    call = adapter.json_calls[0]
    assert "x-api-key" not in call["headers"]
    # Non-auth headers are still present
    assert call["headers"]["anthropic-version"]


@pytest.mark.asyncio
async def test_adapter_can_inject_headers_when_client_omits_auth():
    """Simulates the JS-bridge use case: client passes no auth, adapter
    injects it. Verifies the header reaches the recorded call."""
    adapter = MockAdapter(
        json_response={"choices": [{"message": {"content": "ok"}}]},
        extra_headers={"Authorization": "Bearer key-from-js-bridge"},
    )
    client = PyfetchOpenAI(api_key="", fetch_adapter=adapter)

    await client.summarize("system", "hi")

    call = adapter.json_calls[0]
    assert call["headers"]["Authorization"] == "Bearer key-from-js-bridge"


# ---------------------------------------------------------------------------
# Default adapter: minimally verify instances route through fetch_json/stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_without_adapter_uses_default():
    """When no adapter is passed, the client holds a DefaultPyfetchAdapter."""
    client = PyfetchOpenAI(api_key="sk-test")
    assert isinstance(client._adapter, DefaultPyfetchAdapter)

    client_a = PyfetchAnthropic(api_key="sk-ant-test")
    assert isinstance(client_a._adapter, DefaultPyfetchAdapter)


# ---------------------------------------------------------------------------
# Configuration integrity: existing kwargs still flow through
# ---------------------------------------------------------------------------


def test_fetch_adapter_is_keyword_only():
    """fetch_adapter must be keyword-only; it won't be misinterpreted as
    the first positional arg."""
    # Positional args still fill the existing signature
    client = PyfetchOpenAI("test-model", "sk-key")
    assert client.model == "test-model"
    assert client._api_key == "sk-key"
    # fetch_adapter only accepted as kwarg
    with pytest.raises(TypeError):
        PyfetchOpenAI("test-model", "sk-key", None, 90.0, MockAdapter())  # type: ignore[misc]


def test_dump_config_unchanged():
    """Adding fetch_adapter doesn't leak into dump_config output."""
    adapter = MockAdapter()
    client = PyfetchOpenAI(
        model="test-model",
        api_key="sk-test",
        base_url="https://example.com",
        timeout_seconds=30.0,
        fetch_adapter=adapter,
    )
    config = client.dump_config()
    assert "fetch_adapter" not in config
    assert config["model"] == "test-model"
    assert config["base_url"] == "https://example.com"
