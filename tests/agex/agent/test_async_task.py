import asyncio

import pytest

from agex.agent import Agent
from agex.agent.datatypes import TaskFail
from agex.llm.dummy_client import Dummy
from tests.agex._emissions import make_response


@pytest.mark.asyncio
async def test_async_task_execution():
    """Test standard async task execution."""
    responses = [
        make_response(thinking="Thinking...", code="task_success('async_success')")
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

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
        make_response(
            title="My Title",
            thinking="Thinking...",
            code="task_success('stream_success')",
        )
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

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
    responses = [make_response(thinking="T", code="task_success('done')")]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

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
async def test_async_code_calling_async_function():
    """
    Test that async sandbox code can await async registered functions.
    """
    responses = [
        make_response(
            thinking="Calling async user function",
            code="res = await my_async_fn(5)\ntask_success(res)",
        )
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

    @a.fn
    async def my_async_fn(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    @a.task
    async def mixed_task() -> int:
        """Mixed task."""
        pass

    result = await mixed_task()
    assert result == 10


@pytest.mark.asyncio
async def test_async_task_failure_handling():
    """Test that task_fail works in async task."""
    responses = [make_response(thinking="Failing", code="task_fail('failed')")]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

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

    responses = [make_response(thinking="Need info", code="task_clarify('what is x?')")]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

    @a.task
    async def clarify_task() -> str:
        """Clarify task."""
        pass

    with pytest.raises(TaskClarify) as excinfo:
        await clarify_task()

    assert "what is x?" in str(excinfo.value)


@pytest.mark.asyncio
async def test_async_multi_iteration_via_print():
    """Multi-turn task: turn 1 emits a print (no terminator), turn 2 finishes.

    Under the stateless contract, namespace doesn't carry between
    emissions — multi-turn flow is driven by event log + filesystem,
    not by leftover variables.
    """
    responses = [
        make_response(thinking="First step", code="print('looking around')"),
        make_response(thinking="Second step", code="task_success(2)"),
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

    @a.task
    async def multi_turn_task() -> int:
        """Multi-turn task."""
        pass

    result = await multi_turn_task()
    assert result == 2
    assert len(client.all_events) == 2  # Two LLM calls


@pytest.mark.asyncio
async def test_async_task_error_recovery():
    """Test that evaluation errors are shown to agent and allow recovery."""
    responses = [
        make_response(thinking="Bad code", code="undefined_var"),  # Causes NameError
        make_response(thinking="Fixed", code="task_success('recovered')"),
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

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
        make_response(thinking="Using setup var", code="task_success(setup_value * 2)")
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

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
        make_response(thinking="Step", code="x = 1"),
        make_response(thinking="Step", code="x = 2"),
        make_response(thinking="Step", code="x = 3"),
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client, max_iterations=2)

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
        make_response(
            thinking="Calling async helper",
            code="result = await async_helper(5)\ntask_success(result)",
        )
    ]
    client = Dummy(responses=responses)
    agent = Agent(llm=client)

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


# =============================================================================
# Async Bridge Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_async_fn_error_propagation():
    """Test that errors from async registered functions propagate correctly to agent."""
    responses = [
        # First attempt calls the async fn which raises
        make_response(
            thinking="Calling risky function",
            code="result = await risky_async_fn()",
        ),
        # Agent sees the error and recovers
        make_response(
            thinking="Got error, using fallback",
            code="task_success('recovered_from_async_error')",
        ),
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

    @a.fn
    async def risky_async_fn() -> str:
        """Async function that raises."""
        await asyncio.sleep(0.001)
        raise ValueError("Async function failed!")

    @a.task
    async def error_handling_task() -> str:
        """Task that handles async errors."""
        pass

    result = await error_handling_task()
    assert result == "recovered_from_async_error"
    assert len(client.all_events) == 2  # Two LLM calls


@pytest.mark.asyncio
async def test_async_fn_returns_none():
    """Test async function returning None is handled correctly."""
    responses = [
        make_response(
            thinking="Calling async fn",
            code="result = await async_void_fn()\ntask_success('done' if result is None else 'unexpected')",
        )
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

    @a.fn
    async def async_void_fn() -> None:
        """Async function that returns None."""
        await asyncio.sleep(0.001)
        return None

    @a.task
    async def none_handling_task() -> str:
        """Task that handles None return."""
        pass

    result = await none_handling_task()
    assert result == "done"


@pytest.mark.asyncio
async def test_async_fn_complex_return_type():
    """Test async function returning complex types."""
    responses = [
        make_response(
            thinking="Calling async fn",
            code="data = await fetch_complex_data()\ntask_success(data['items'][0])",
        )
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

    @a.fn
    async def fetch_complex_data() -> dict:
        """Async function that returns complex data."""
        await asyncio.sleep(0.001)
        return {"items": ["first", "second"], "count": 2}

    @a.task
    async def complex_data_task() -> str:
        """Task that uses complex data."""
        pass

    result = await complex_data_task()
    assert result == "first"


@pytest.mark.asyncio
async def test_multiple_async_fn_calls():
    """Test multiple async function calls in sequence."""
    responses = [
        make_response(
            thinking="Calling multiple async functions",
            code="a = await async_add(1)\nb = await async_add(a)\nc = await async_add(b)\ntask_success(c)",
        )
    ]
    client = Dummy(responses=responses)
    agent = Agent(llm=client)

    @agent.fn
    async def async_add(x: int) -> int:
        """Async function that adds 10."""
        await asyncio.sleep(0.001)
        return x + 10

    @agent.task
    async def chain_task() -> int:
        """Task that chains async calls."""
        pass

    result = await chain_task()
    assert result == 31  # 1 + 10 + 10 + 10


@pytest.mark.asyncio
async def test_async_fn_with_exception_type_preserved():
    """Test that specific exception types from async functions are preserved."""
    responses = [
        make_response(
            thinking="Calling fn that raises KeyError",
            code="""
try:
    result = await async_key_error_fn()
except KeyError as e:
    task_success(f'caught_keyerror:{e}')
""",
        ),
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

    @a.fn
    async def async_key_error_fn() -> str:
        """Async function that raises KeyError."""
        await asyncio.sleep(0.001)
        raise KeyError("missing_key")

    @a.task
    async def exception_type_task() -> str:
        """Task that catches specific exception."""
        pass

    result = await exception_type_task()
    assert "caught_keyerror" in result
    assert "missing_key" in result


def test_sync_task_async_fn_error_surfaces():
    """Test that calling async fn from sync task produces clear error to agent.

    Calling an async function without ``await`` returns a coroutine object
    rather than a result.  The agent sees unexpected output and recovers.
    The sandbox code explicitly closes the coroutine to avoid RuntimeWarning.
    """
    responses = [
        # Agent tries to call async fn — close the coroutine to avoid GC warning
        make_response(
            thinking="Calling async function",
            code="result = async_fn_not_available()\nif hasattr(result, 'close'): result.close()",
        ),
        # Agent sees error and uses fallback
        make_response(
            thinking="Got error about async fn, using fallback",
            code="task_success('used_sync_fallback')",
        ),
    ]
    client = Dummy(responses=responses)
    a = Agent(llm=client)

    @a.fn
    async def async_fn_not_available() -> str:
        """Async function that can't be called from sync task."""
        await asyncio.sleep(0.001)
        return "async_result"

    @a.task
    def sync_task() -> str:
        """Sync task that tries to call async fn."""
        pass

    # Agent should see error in stdout and recover
    result = sync_task()
    assert result == "used_sync_fallback"
    assert len(client.all_events) == 2  # Two LLM calls (error + recovery)
