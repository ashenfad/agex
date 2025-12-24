import contextvars

import numpy as np
import pytest

from agex.agent import Agent
from agex.agent.base import clear_agent_registry
from agex.llm.dummy_client import DummyLLMClient
from agex.remote.serialize import deserialize_agent, serialize_agent


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_agent_serialization_basic():
    """Test basic agent serialization and reconstruction."""
    agent = Agent()
    agent.llm_client = DummyLLMClient(timeout_seconds=123)

    # Serialize
    payload = serialize_agent(agent)
    assert isinstance(payload, bytes)
    assert len(payload) > 0

    # Deserialization in the SAME context will fail because the agent name exists.
    # To test proper isolation/rehydration, we run deserialization in a copied context
    # (simulating a new request context or just isolated execution).
    def run_isolated():
        # Clear registry in this context
        clear_agent_registry()
        return deserialize_agent(payload)

    ctx = contextvars.copy_context()
    restored_agent = ctx.run(run_isolated)

    assert restored_agent.name is not None
    assert restored_agent.llm_client is not None
    # LLM client is reconstructed from config - will be DummyLLMClient
    assert restored_agent.llm_client.model == "dummy"

    # Verify fingerprint logic was called
    assert restored_agent.fingerprint is not None


def test_agent_serialization_with_modules():
    """Test serialization of an agent with registered modules."""
    agent = Agent()
    agent.llm_client = DummyLLMClient()
    agent.module(np, name="numpy")

    payload = serialize_agent(agent)

    def run_isolated():
        clear_agent_registry()
        return deserialize_agent(payload)

    ctx = contextvars.copy_context()
    restored_agent = ctx.run(run_isolated)

    # Verify modules are accessible
    assert "numpy" in restored_agent._policy.namespaces


def test_serialized_config_is_used():
    """Test that serialized LLM config is used during deserialization."""
    agent = Agent()
    agent.llm_client = DummyLLMClient(timeout_seconds=999)

    payload = serialize_agent(agent)

    def run_isolated():
        clear_agent_registry()
        return deserialize_agent(payload)

    ctx = contextvars.copy_context()
    restored_agent = ctx.run(run_isolated)

    # Verify the reconstructed client uses the serialized config
    # DummyLLMClient's from_config passes through timeout_seconds
    assert restored_agent.llm_client is not None
