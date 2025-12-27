"""Tests for server app factory and endpoints."""

import base64

import cloudpickle
import pytest
from fastapi.testclient import TestClient

from agex import Agent
from agex.agent.base import clear_agent_registry
from agex.llm.dummy_client import Dummy


def encode_args_kwargs(args=(), kwargs=None):
    """Helper to encode args/kwargs for HTTP payload."""
    if kwargs is None:
        kwargs = {}
    return {
        "args": base64.b64encode(cloudpickle.dumps(args)).decode("utf-8"),
        "kwargs": base64.b64encode(cloudpickle.dumps(kwargs)).decode("utf-8"),
    }


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


@pytest.fixture
def mock_llm():
    return Dummy()


@pytest.fixture
def app(tmp_path):
    """Create test app."""
    from agex.server import create_app

    return create_app(state_dir=str(tmp_path))


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_execute_missing_task(client, mock_llm):
    """Test execute with non-existent task."""
    # Create a fresh agent for this test
    clear_agent_registry()  # Clear before creating
    agent = Agent()
    agent.llm = mock_llm
    from agex.host import serialize_agent

    payload = {
        "agent_payload": base64.b64encode(serialize_agent(agent)).decode("utf-8"),
        "task_name": "nonexistent_task",
        **encode_args_kwargs(),
    }

    # Clear registry again so deserialization doesn't collide
    clear_agent_registry()

    response = client.post("/execute", json=payload)
    assert response.status_code == 200

    # Parse SSE response
    content = response.text
    assert "error" in content
    assert "not found" in content.lower()


def test_execute_with_session(client, mock_llm):
    """Test execute with session parameter."""
    # Create a fresh agent for this test
    clear_agent_registry()
    agent = Agent()
    agent.llm = mock_llm
    from agex.host import serialize_agent

    payload = {
        "agent_payload": base64.b64encode(serialize_agent(agent)).decode("utf-8"),
        "task_name": "some_task",
        **encode_args_kwargs(),
        "session": "test_session",  # Valid session
    }

    # Clear registry again so deserialization doesn't collide
    clear_agent_registry()

    response = client.post("/execute", json=payload)
    assert response.status_code == 200

    # Task doesn't exist, so we get a "not found" error
    content = response.text
    assert "error" in content
    assert "not found" in content.lower()
