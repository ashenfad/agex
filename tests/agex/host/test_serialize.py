import contextvars

import numpy as np
import pytest

from agex.agent import Agent
from agex.agent.base import clear_agent_registry
from agex.host import deserialize_agent, prepare_agent, serialize_agent
from agex.llm.dummy_client import Dummy


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestSerializeAgent:
    """Tests for basic serialize/deserialize."""

    def test_serialize_returns_bytes(self):
        """Test that serialize_agent returns bytes."""
        agent = Agent()
        agent.llm = Dummy()

        payload = serialize_agent(agent)
        assert isinstance(payload, bytes)
        assert len(payload) > 0

    def test_deserialize_returns_agent(self):
        """Test that deserialize_agent returns an agent-like object."""
        agent = Agent()
        agent.llm = Dummy()

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            return deserialize_agent(payload)

        ctx = contextvars.copy_context()
        restored = ctx.run(run_isolated)

        assert restored.name is not None
        assert hasattr(restored, "_tasks")

    def test_deserialize_preserves_modules(self):
        """Test serialization preserves registered modules."""
        agent = Agent()
        agent.llm = Dummy()
        agent.module(np, name="numpy")

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            return deserialize_agent(payload)

        ctx = contextvars.copy_context()
        restored = ctx.run(run_isolated)

        assert "numpy" in restored._policy.namespaces


class TestPrepareAgent:
    """Tests for prepare_agent (full remote execution setup)."""

    def test_prepare_rehydrates_llm(self):
        """Test that prepare_agent rehydrates the LLM client."""
        agent = Agent()
        agent.llm = Dummy(timeout_seconds=123)

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            return prepare_agent(payload)

        ctx = contextvars.copy_context()
        restored = ctx.run(run_isolated)

        # LLM is rehydrated from config
        assert restored.llm is not None
        assert restored.llm.model == "dummy"

    def test_prepare_forces_local_host(self):
        """Test that prepare_agent forces Local host for server-side execution.

        This prevents infinite loops where sub-agents would route back to
        HTTP hosts. Sub-agents always execute locally within their parent's
        execution context.
        """
        from agex.host import HTTP, Local

        agent = Agent()
        agent.llm = Dummy()
        agent._host = HTTP(url="http://remote-server:8000")

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            return prepare_agent(payload)

        ctx = contextvars.copy_context()
        restored = ctx.run(run_isolated)

        # Host is forced to Local for server-side execution
        assert isinstance(restored._host, Local)

    def test_prepare_updates_fingerprint(self):
        """Test that prepare_agent updates the agent fingerprint."""
        agent = Agent()
        agent.llm = Dummy()

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            return prepare_agent(payload)

        ctx = contextvars.copy_context()
        restored = ctx.run(run_isolated)

        # Fingerprint is set
        assert restored.fingerprint is not None

    def test_prepare_preserves_llm_config(self):
        """Test that prepare_agent preserves LLM configuration."""
        agent = Agent()
        agent.llm = Dummy(timeout_seconds=999)

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            return prepare_agent(payload)

        ctx = contextvars.copy_context()
        restored = ctx.run(run_isolated)

        # Config is preserved through rehydration
        assert restored.llm is not None
