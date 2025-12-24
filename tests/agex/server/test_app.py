"""Tests for server app factory and endpoints."""

import base64

import pytest
from fastapi.testclient import TestClient

from agex import Agent
from agex.agent.base import clear_agent_registry
from agex.llm import LLMClient


class MockClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, model="mock", timeout_seconds=60, **kwargs):
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def model(self):
        return self._model

    @property
    def provider_name(self):
        return "mock"

    def dump_config(self):
        return {
            "provider": "mock",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_config(cls, config):
        return cls(model=config["model"], timeout_seconds=config["timeout_seconds"])

    def complete(self, *args, **kwargs):
        pass

    def summarize(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


@pytest.fixture
def mock_llm_client():
    return MockClient()


@pytest.fixture
def app(mock_llm_client, tmp_path):
    """Create test app with mock LLM client."""
    from agex.server import create_app

    return create_app(llm_client=mock_llm_client, state_base_path=str(tmp_path))


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_execute_missing_task(client, mock_llm_client):
    """Test execute with non-existent task."""
    # Create a fresh agent for this test
    clear_agent_registry()  # Clear before creating
    agent = Agent()
    agent.llm_client = mock_llm_client
    from agex.remote import serialize_agent

    payload = {
        "agent_payload": base64.b64encode(serialize_agent(agent)).decode("utf-8"),
        "task_name": "nonexistent_task",
        "args": [],
        "kwargs": {},
    }

    # Clear registry again so deserialization doesn't collide
    clear_agent_registry()

    response = client.post("/execute", json=payload)
    assert response.status_code == 200

    # Parse SSE response
    content = response.text
    assert "error" in content
    assert "not found" in content.lower()


def test_execute_invalid_state_uri(client, mock_llm_client):
    """Test execute with invalid state URI."""
    # Create a fresh agent for this test
    clear_agent_registry()
    agent = Agent()
    agent.llm_client = mock_llm_client
    from agex.remote import serialize_agent

    payload = {
        "agent_payload": base64.b64encode(serialize_agent(agent)).decode("utf-8"),
        "task_name": "some_task",
        "args": [],
        "kwargs": {},
        "state_uri": "disk://../etc/passwd",  # Path traversal attempt
    }

    # Clear registry again so deserialization doesn't collide
    clear_agent_registry()

    response = client.post("/execute", json=payload)
    assert response.status_code == 200

    # Parse SSE response - we expect an error about alphanumeric chars
    content = response.text
    assert "error" in content
    assert "alphanumeric" in content.lower()
