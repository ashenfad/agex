import asyncio

import pytest

from agex.agent import Agent
from agex.agent.datatypes import TaskFail
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import DummyLLMClient


@pytest.mark.asyncio
async def test_async_task_execution():
    """Test standard async task execution."""
    responses = [
        LLMResponse(thinking="Thinking...", code="task_success('async_success')")
    ]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    @a.task
    async def my_task(val: int) -> str:
        """Test task."""
        pass

    result = await my_task(10)
    assert result == "async_success"
    # Verify events
    assert len(client.all_events) == 1  # 1 LLM call


@pytest.mark.asyncio
async def test_async_streaming():
    """Test streaming with async task."""
    responses = [
        LLMResponse(
            title="My Title",
            thinking="Thinking...",
            code="task_success('stream_success')",
        )
    ]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    tokens = []

    async def token_handler(token):
        tokens.append(token)
        # Verify we can await inside handler
        await asyncio.sleep(0.0001)

    @a.task
    async def stream_task() -> str:
        """Stream task."""
        pass

    result = await stream_task(on_token=token_handler)
    assert result == "stream_success"

    types = [t.type for t in tokens]
    assert "title" in types
    assert "thinking" in types
    assert "python" in types


@pytest.mark.asyncio
async def test_async_event_handler():
    """Test async event handler."""
    responses = [LLMResponse(thinking="T", code="task_success('done')")]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    events = []

    async def event_handler(evt):
        events.append(evt)
        await asyncio.sleep(0.0001)

    @a.task
    async def handler_task() -> str:
        """Handler task."""
        pass

    await handler_task(on_event=event_handler)
    assert len(events) > 0


@pytest.mark.asyncio
async def test_sync_code_calling_async_function_in_async_task():
    """
    Test that sync code (evaluated in thread) can transparently call
    async functions (bridged to main loop).
    """
    responses = [
        LLMResponse(
            thinking="Calling async user function",
            code="res = my_async_fn(5); task_success(res)",
        )
    ]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    # Register an async user function
    @a.fn
    async def my_async_fn(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    @a.task
    async def mixed_task() -> int:
        """Mixed task."""
        pass

    # When the agent calls 'my_async_fn(5)' in the generated code (which is sync),
    # it receives a coroutine. The CallEvaluator should detect this, bridge it
    # to the main loop, wait for result, and return 10.
    result = await mixed_task()
    assert result == 10


@pytest.mark.asyncio
async def test_async_task_failure_handling():
    """Test that task_fail works in async task."""
    responses = [LLMResponse(thinking="Failing", code="task_fail('failed')")]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    @a.task
    async def fail_task():
        """Fail task."""
        pass

    with pytest.raises(TaskFail) as excinfo:
        await fail_task()

    assert "failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_async_task_clarify_handling():
    """Test that task_clarify works in async task."""
    from agex.agent.datatypes import TaskClarify

    responses = [LLMResponse(thinking="Need info", code="task_clarify('what is x?')")]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    @a.task
    async def clarify_task() -> str:
        """Clarify task."""
        pass

    with pytest.raises(TaskClarify) as excinfo:
        await clarify_task()

    assert "what is x?" in str(excinfo.value)


@pytest.mark.asyncio
async def test_async_task_continue():
    """Test that task_continue works in async task (multiple iterations)."""
    responses = [
        LLMResponse(thinking="First step", code="x = 1; task_continue()"),
        LLMResponse(thinking="Second step", code="task_success(x + 1)"),
    ]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    @a.task
    async def continue_task() -> int:
        """Continue task."""
        pass

    result = await continue_task()
    assert result == 2
    assert len(client.all_events) == 2  # Two LLM calls


@pytest.mark.asyncio
async def test_async_task_error_recovery():
    """Test that evaluation errors are shown to agent and allow recovery."""
    responses = [
        LLMResponse(thinking="Bad code", code="undefined_var"),  # Causes NameError
        LLMResponse(thinking="Fixed", code="task_success('recovered')"),
    ]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    @a.task
    async def error_task() -> str:
        """Error task."""
        pass

    result = await error_task()
    assert result == "recovered"
    assert len(client.all_events) == 2


@pytest.mark.asyncio
async def test_async_task_with_setup():
    """Test setup code execution in async task."""
    responses = [
        LLMResponse(thinking="Using setup var", code="task_success(setup_value * 2)")
    ]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client)

    @a.task(setup="setup_value = 21")
    async def setup_task() -> int:
        """Setup task."""
        pass

    result = await setup_task()
    assert result == 42


@pytest.mark.asyncio
async def test_async_task_timeout():
    """Test that max_iterations causes TaskTimeout in async task."""
    from agex.agent.datatypes import TaskTimeout

    # Agent never calls task_success
    responses = [
        LLMResponse(thinking="Step", code="x = 1"),
        LLMResponse(thinking="Step", code="x = 2"),
        LLMResponse(thinking="Step", code="x = 3"),
    ]
    client = DummyLLMClient(responses=responses)
    a = Agent(llm_client=client, max_iterations=2)

    @a.task
    async def timeout_task() -> int:
        """Timeout task."""
        pass

    with pytest.raises(TaskTimeout):
        await timeout_task()


@pytest.mark.asyncio
async def test_async_recursive_task():
    """Test async task calling a registered async function."""
    responses = [
        LLMResponse(
            thinking="Calling async helper",
            code="result = async_helper(5); task_success(result)",
        )
    ]
    client = DummyLLMClient(responses=responses)
    agent = Agent(llm_client=client)

    @agent.fn
    async def async_helper(x: int) -> int:
        """Async helper function."""
        await asyncio.sleep(0.001)
        return x * 3

    @agent.task
    async def main_task() -> int:
        """Main task."""
        pass

    result = await main_task()
    assert result == 15  # 5 * 3
