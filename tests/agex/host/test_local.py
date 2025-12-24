"""Tests for Local host implementation."""

import pytest

from agex import Agent
from agex.agent.base import clear_agent_registry
from agex.host import Local
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_local_is_default_host():
    """Test that Local is the default host for agents."""
    agent = Agent()
    assert isinstance(agent._host, Local)


def test_local_execute_runs_task():
    """Test that Local.execute runs the task loop."""
    llm = Dummy(responses=[LLMResponse(thinking="done", code="task_success(42)")])
    agent = Agent(llm=llm)

    @agent.task
    def get_answer() -> int:
        """Return 42."""
        pass

    # Task should run via Local host
    result = get_answer()
    assert result == 42


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_local_aexecute_runs_async_task(anyio_backend):
    """Test that Local.aexecute runs async tasks."""
    llm = Dummy(responses=[LLMResponse(thinking="done", code="task_success(42)")])
    agent = Agent(llm=llm)

    @agent.task
    async def get_answer() -> int:
        """Return 42."""
        pass

    result = await get_answer()
    assert result == 42
