"""
Integration tests for on_conflict task parameter with Versioned state.

These tests verify automatic merge/retry behavior when concurrent tasks
modify the same Versioned state.
"""

from kvgit import Staged, VersionedKV

from agex import Agent
from agex.llm import Dummy
from agex.llm.core import LLMResponse
from agex.state import _agex_decoder, _agex_encoder, connect_state, kv


def test_task_merges_on_success():
    """Test that successful task with versioned state completes correctly."""
    responses = [
        LLMResponse(
            thinking="I'll store a value and complete.",
            code='x = 42\ntask_success("done")',
        )
    ]
    llm = Dummy(responses=responses)

    config = connect_state(type="versioned", storage="memory")
    agent = Agent(max_iterations=2, llm=llm, state=config)

    @agent.task
    def simple_task() -> str:
        """Complete a simple task."""
        pass

    result = simple_task()
    assert result == "done"


def test_task_retry_on_conflict():
    """Test that on_conflict='retry' doesn't break normal operation."""
    responses = [
        LLMResponse(
            thinking="First attempt.",
            code='task_success("attempt1")',
        ),
        LLMResponse(
            thinking="Second attempt after retry.",
            code='task_success("attempt2")',
        ),
    ]
    llm = Dummy(responses=responses)

    config = connect_state(type="versioned", storage="memory")
    agent = Agent(max_iterations=2, llm=llm, state=config)

    @agent.task(on_conflict="retry", max_conflict_retries=2)
    def retry_task() -> str:
        """Task that may need retry."""
        pass

    # Test that the retry mechanism doesn't break normal operation
    result = retry_task()
    assert result == "attempt1"


def test_task_abandon_on_conflict():
    """Test that on_conflict='abandon' works in normal case."""
    responses = [
        LLMResponse(
            thinking="Background task work.",
            code='x = 1\ntask_success("background_result")',
        )
    ]
    llm = Dummy(responses=responses)

    config = connect_state(type="versioned", storage="memory")
    agent = Agent(max_iterations=2, llm=llm, state=config)

    @agent.task(on_conflict="abandon")
    def background_task() -> str:
        """Background task that can be abandoned."""
        pass

    result = background_task()
    assert result == "background_result"


def test_task_without_versioned_state_ignores_on_conflict():
    """Test that on_conflict is ignored when not using Versioned state."""
    responses = [
        LLMResponse(
            thinking="Simple task.",
            code="task_success(42)",
        )
    ]
    llm = Dummy(responses=responses)
    agent = Agent(max_iterations=2, llm=llm)

    @agent.task(on_conflict="retry")
    def stateless_task() -> int:
        """Task without state."""
        pass

    # Should work fine without state (ephemeral)
    result = stateless_task()
    assert result == 42


def test_task_with_multiple_snapshots_merges_all():
    """Test that multiple snapshots within a task complete correctly."""
    responses = [
        LLMResponse(
            thinking="First step.",
            code='step1 = "done"\ntask_continue()',
        ),
        LLMResponse(
            thinking="Second step.",
            code='step2 = "done"\ntask_continue()',
        ),
        LLMResponse(
            thinking="Final step.",
            code='task_success("complete")',
        ),
    ]
    llm = Dummy(responses=responses)

    config = connect_state(type="versioned", storage="memory")
    agent = Agent(max_iterations=5, llm=llm, state=config)

    @agent.task
    def multi_step_task() -> str:
        """Task with multiple steps."""
        pass

    result = multi_step_task()
    assert result == "complete"


def test_concurrent_tasks_with_actual_conflict():
    """Test real concurrent conflict with threading - verify retry works.

    This test requires a shared KV store to simulate conflicts.
    We access the session cache directly to inject a shared store.
    """
    import threading
    import time

    store = kv.Memory()
    results = {}
    errors = {}

    def _make_shared(kv_store):
        return Staged(
            VersionedKV(kv_store), encoder=_agex_encoder, decoder=_agex_decoder
        )

    # Create two agents that will run concurrently with shared store
    agent1_responses = [
        LLMResponse(
            thinking="Agent 1 working.",
            code='data1 = "agent1_data"\ntask_success("agent1_done")',
        ),
        LLMResponse(
            thinking="Agent 1 retrying after conflict.",
            code='data1_retry = "agent1_retry"\ntask_success("agent1_retried")',
        ),
    ]
    config1 = connect_state(type="versioned", storage="memory")
    agent1 = Agent(
        max_iterations=2, llm=Dummy(responses=agent1_responses), state=config1
    )
    # Inject shared store for conflict testing
    agent1._host._session_cache["versioned:default"] = _make_shared(store)

    agent2_responses = [
        LLMResponse(
            thinking="Agent 2 working.",
            code='data2 = "agent2_data"\ntask_success("agent2_done")',
        ),
        LLMResponse(
            thinking="Agent 2 retrying after conflict.",
            code='data2_retry = "agent2_retry"\ntask_success("agent2_retried")',
        ),
    ]
    config2 = connect_state(type="versioned", storage="memory")
    agent2 = Agent(
        max_iterations=2, llm=Dummy(responses=agent2_responses), state=config2
    )
    # Inject shared store for conflict testing
    agent2._host._session_cache["versioned:default"] = _make_shared(store)

    @agent1.task(on_conflict="retry", max_conflict_retries=2)
    def task1() -> str:
        """Task 1."""
        pass

    @agent2.task(on_conflict="retry", max_conflict_retries=2)
    def task2() -> str:
        """Task 2."""
        pass

    def run_task1():
        try:
            results["task1"] = task1()
        except Exception as e:
            errors["task1"] = e

    def run_task2():
        time.sleep(0.05)  # Small delay to let task1 start first
        try:
            results["task2"] = task2()
        except Exception as e:
            errors["task2"] = e

    t1 = threading.Thread(target=run_task1)
    t2 = threading.Thread(target=run_task2)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Unexpected errors: {errors}"
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    # At least one should have retried
    retried = (
        "agent1_retried" in results.values() or "agent2_retried" in results.values()
    )
    assert retried, f"Expected at least one retry, but got results: {results}"


def test_concurrent_abandon_strategy():
    """Test that abandon strategy doesn't raise errors on potential conflicts.

    This test requires a shared KV store to simulate conflicts.
    """
    import threading

    store = kv.Memory()
    results = {}

    def _make_shared(kv_store):
        return Staged(
            VersionedKV(kv_store), encoder=_agex_encoder, decoder=_agex_decoder
        )

    agent1_responses = [
        LLMResponse(
            thinking="Background task 1.",
            code='bg1 = "data"\ntask_success("bg1_done")',
        )
    ]
    config1 = connect_state(type="versioned", storage="memory")
    agent1 = Agent(
        max_iterations=2, llm=Dummy(responses=agent1_responses), state=config1
    )
    agent1._host._session_cache["versioned:default"] = _make_shared(store)

    agent2_responses = [
        LLMResponse(
            thinking="Background task 2.",
            code='bg2 = "data"\ntask_success("bg2_done")',
        )
    ]
    config2 = connect_state(type="versioned", storage="memory")
    agent2 = Agent(
        max_iterations=2, llm=Dummy(responses=agent2_responses), state=config2
    )
    agent2._host._session_cache["versioned:default"] = _make_shared(store)

    @agent1.task(on_conflict="abandon")
    def bg_task1() -> str:
        """Background task 1."""
        pass

    @agent2.task(on_conflict="abandon")
    def bg_task2() -> str:
        """Background task 2."""
        pass

    def run_bg1():
        results["bg1"] = bg_task1()

    def run_bg2():
        results["bg2"] = bg_task2()

    t1 = threading.Thread(target=run_bg1)
    t2 = threading.Thread(target=run_bg2)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert "bg1" in results
    assert "bg2" in results
    # In case of conflict, one might return None (abandoned)
    assert results["bg1"] in ["bg1_done", None]
    assert results["bg2"] in ["bg2_done", None]
