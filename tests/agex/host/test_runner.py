"""Tests for remote task execution runner."""

import contextvars

import pytest

from agex.agent import Agent
from agex.agent.base import clear_agent_registry
from agex.host import execute_task, prepare_agent, run_remote_task, serialize_agent
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestExecuteTask:
    """Tests for execute_task function."""

    def test_execute_task_runs_task(self):
        """Test that execute_task runs a task and returns result."""
        llm = Dummy(responses=[LLMResponse(thinking="done", code="task_success(42)")])
        agent = Agent(llm=llm)

        @agent.task
        def get_answer() -> int:
            """Return 42."""
            pass

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            prepared = prepare_agent(payload)
            return execute_task(prepared, "get_answer")

        ctx = contextvars.copy_context()
        result = ctx.run(run_isolated)

        assert result == 42

    def test_execute_task_with_kwargs(self):
        """Test execute_task with keyword arguments."""
        llm = Dummy(responses=[LLMResponse(thinking="done", code="task_success(100)")])
        agent = Agent(llm=llm)

        @agent.task
        def process_data(data: str) -> int:
            """Process the data and return a count."""
            pass

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            prepared = prepare_agent(payload)
            return execute_task(prepared, "process_data", kwargs={"data": "test input"})

        ctx = contextvars.copy_context()
        result = ctx.run(run_isolated)

        assert result == 100

    def test_execute_task_unknown_task_raises(self):
        """Test that execute_task raises for unknown task."""
        agent = Agent(llm=Dummy())

        @agent.task
        def known_task() -> int:
            """A known task."""
            pass

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            prepared = prepare_agent(payload)
            return execute_task(prepared, "unknown_task")

        ctx = contextvars.copy_context()
        with pytest.raises(KeyError, match="unknown_task"):
            ctx.run(run_isolated)

    def test_execute_task_with_callbacks(self):
        """Test execute_task calls event/token callbacks."""
        llm = Dummy(responses=[LLMResponse(thinking="done", code="task_success(1)")])
        agent = Agent(llm=llm)

        @agent.task
        def callback_task() -> int:
            """A task for callback testing."""
            pass

        payload = serialize_agent(agent)
        events_received = []

        def run_isolated():
            clear_agent_registry()
            prepared = prepare_agent(payload)
            return execute_task(
                prepared,
                "callback_task",
                on_event=lambda e: events_received.append(e),
            )

        ctx = contextvars.copy_context()
        result = ctx.run(run_isolated)

        assert result == 1
        assert len(events_received) > 0  # Should have at least TaskStartEvent


class TestRunRemoteTask:
    """Tests for run_remote_task convenience function."""

    def test_run_remote_task_combines_prepare_and_execute(self):
        """Test run_remote_task does everything in one call."""
        llm = Dummy(responses=[LLMResponse(thinking="ok", code="task_success(99)")])
        agent = Agent(llm=llm)

        @agent.task
        def compute() -> int:
            """Compute something."""
            pass

        payload = serialize_agent(agent)

        def run_isolated():
            clear_agent_registry()
            return run_remote_task(payload, "compute")

        ctx = contextvars.copy_context()
        result = ctx.run(run_isolated)

        assert result == 99
