import contextvars

import numpy as np
import pytest

from agex.agent import Agent
from agex.agent.base import clear_agent_registry
from agex.llm import LLMClient
from agex.remote.serialize import deserialize_agent, serialize_agent


class MockClient(LLMClient):
    """Concrete mock client for testing."""

    def __init__(self, model="mock-model", timeout_seconds=60, **kwargs):
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


def test_agent_serialization_basic():
    """Test basic agent serialization and reconstruction."""
    agent = Agent()
    agent.llm_client = MockClient(model="serialization-test", timeout_seconds=123)

    # Serialize
    payload = serialize_agent(agent)
    assert isinstance(payload, bytes)
    assert len(payload) > 0

    # Deserialization in the SAME context will fail because the agent name exists.
    # To test proper isolation/rehydration, we run deserialization in a copied context
    # (simulating a new request context or just isolated execution).
    # Since we use ContextVars for registry, a new context should have its own registry copy?
    # ContextVar.set() affects the current context. copy_context() copies values.
    # Wait, copy_context() COPIES the values. So the registry will still contain the agent!
    # We need a CLEAN context.

    # We can use a helper to run in a clean context context.
    def run_isolated():
        # Clear registry in this context
        clear_agent_registry()
        return deserialize_agent(
            payload,
            llm_client=MockClient(model="serialization-test", timeout_seconds=123),
        )

    ctx = contextvars.copy_context()
    restored_agent = ctx.run(run_isolated)

    assert restored_agent.name is not None
    assert restored_agent.llm_client is not None
    assert restored_agent.llm_client.model == "serialization-test"
    assert restored_agent.llm_client.timeout_seconds == 123

    # Verify fingerprint logic was called
    assert restored_agent.fingerprint is not None


def test_agent_serialization_with_modules():
    """Test serialization of an agent with registered modules."""
    agent = Agent()
    agent.module(np, name="numpy")

    payload = serialize_agent(agent)

    # Inject client to avoid reconstruction error, run isolated
    def run_isolated():
        clear_agent_registry()
        return deserialize_agent(payload, llm_client=MockClient())

    ctx = contextvars.copy_context()
    restored_agent = ctx.run(run_isolated)

    # Verify modules are accessible (though they aren't directly inspected easily)
    # Ideally we'd compile a program that uses them, but for unit test, check internal state?
    # Agex policy stores imports.
    assert "numpy" in restored_agent._policy.namespaces


def test_manual_client_injection():
    """Test injecting a fresh client during deserialization."""
    agent = Agent()
    agent.llm_client = MockClient(model="original")

    payload = serialize_agent(agent)

    new_client = MockClient(model="injected")

    def run_isolated():
        clear_agent_registry()
        return deserialize_agent(payload, llm_client=new_client)

    ctx = contextvars.copy_context()
    restored_agent = ctx.run(run_isolated)

    assert restored_agent.llm_client.model == "injected"
