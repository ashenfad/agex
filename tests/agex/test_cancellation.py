"""Tests for task cancellation functionality."""

import threading

import pytest

from agex import Agent, TaskCancelled, connect_state, events
from agex.llm import Dummy
from agex.llm.core import LLMResponse
from agex.state import Versioned


def _set_cancel_sentinel(state: Versioned, task_name: str) -> None:
    """Write cancellation sentinel directly to KV store (like cancel() does)."""
    cancel_key = f"__agex_cancel__{task_name}"
    state.set_raw(cancel_key, True)


def _get_cancel_sentinel(state: Versioned, task_name: str) -> bool | None:
    """Read cancellation sentinel directly from KV store."""
    cancel_key = f"__agex_cancel__{task_name}"
    return state.get_raw(cancel_key)


class TestTaskCancellation:
    """Tests for task cancellation via sentinel in state."""

    def test_cancel_sets_sentinel_in_state(self):
        """cancel() writes the sentinel to state."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Working", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """A simple task."""
            pass

        # Run task first to initialize state
        my_task()

        # Call cancel (task not running, but should still write sentinel)
        my_task.cancel()

        # Verify sentinel is in state (directly in KV store)
        state = agent.state()
        assert _get_cancel_sentinel(state, "my_task") is True

    def test_cancel_with_custom_session(self):
        """cancel() respects session parameter."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Working", code='task_success("done")'),
                LLMResponse(thinking="Working", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """A simple task."""
            pass

        # Run on two sessions
        my_task(session="alice")
        my_task(session="bob")

        # Cancel only alice's session
        my_task.cancel(session="alice")

        # Verify sentinel only in alice's state
        alice_state = agent.state(session="alice")
        bob_state = agent.state(session="bob")

        assert _get_cancel_sentinel(alice_state, "my_task") is True
        assert _get_cancel_sentinel(bob_state, "my_task") is None

    def test_task_raises_cancelled_when_sentinel_present(self):
        """Task raises TaskCancelled when sentinel is in state at iteration start."""
        # Use multiple responses - first succeeds, second should be cancelled
        llm = Dummy(
            responses=[
                # First task run - completes normally
                LLMResponse(thinking="First", code='task_success("first")'),
                # Second task run - won't complete due to cancel
                LLMResponse(thinking="Second", code='task_success("second")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def cancellable_task() -> str:
            """A task that can be cancelled."""
            pass

        # First run succeeds
        result = cancellable_task()
        assert result == "first"

        # Set cancellation sentinel before second run (directly in KV store)
        state = agent.state()
        _set_cancel_sentinel(state, "cancellable_task")

        # Second run should raise TaskCancelled
        with pytest.raises(TaskCancelled) as exc_info:
            cancellable_task()

        assert exc_info.value.task_name == "cancellable_task"
        assert "cancelled" in exc_info.value.message.lower()

    def test_cancelled_cleans_up_sentinel(self):
        """When task is cancelled, sentinel is removed from state."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="First", code='task_success("ok")'),
                LLMResponse(thinking="Second", code='task_success("ok")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        # First run to initialize state
        my_task()

        # Set sentinel (directly in KV store)
        state = agent.state()
        _set_cancel_sentinel(state, "my_task")

        # Run should raise but clean up sentinel
        with pytest.raises(TaskCancelled):
            my_task()

        # Sentinel should be gone
        state = agent.state()
        assert _get_cancel_sentinel(state, "my_task") is None

    def test_cancelled_iterations_completed(self):
        """TaskCancelled includes iterations_completed count."""
        # Task that requires multiple iterations
        llm = Dummy(
            responses=[
                # First run - complete
                LLMResponse(thinking="Working", code='task_success("done")'),
                # Second run - will be cancelled at start (0 iterations)
                LLMResponse(thinking="Working", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        # First run
        my_task()

        # Set sentinel before second run (directly in KV store)
        state = agent.state()
        _set_cancel_sentinel(state, "my_task")

        # Second run cancelled at iteration 0
        with pytest.raises(TaskCancelled) as exc_info:
            my_task()

        assert exc_info.value.iterations_completed == 0

    def test_different_tasks_have_separate_sentinels(self):
        """Each task has its own cancel sentinel."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="A", code='task_success("a")'),
                LLMResponse(thinking="B", code='task_success("b")'),
                LLMResponse(thinking="A2", code='task_success("a2")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def task_a() -> str:
            """Task A."""
            pass

        @agent.task
        def task_b() -> str:
            """Task B."""
            pass

        # Run both to initialize
        task_a()
        task_b()

        # Cancel only task_a
        task_a.cancel()

        # task_a should be cancelled
        with pytest.raises(TaskCancelled):
            task_a()

    def test_cancel_with_versioned_disk_state(self, tmp_path):
        """cancel() works with disk-backed versioned state."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Working", code='task_success("done")'),
            ]
        )
        agent = Agent(
            state=connect_state(type="versioned", storage="disk", path=str(tmp_path)),
            llm=llm,
        )

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        # Run to initialize
        my_task()

        # Cancel should work
        my_task.cancel()

        # Verify sentinel persists in disk state
        state = agent.state()
        assert _get_cancel_sentinel(state, "my_task") is True


class TestTaskCancellationAsync:
    """Async-specific cancellation tests."""

    @pytest.mark.asyncio
    async def test_async_task_raises_cancelled(self):
        """Async task raises TaskCancelled when sentinel is present."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="First", code='task_success("first")'),
                LLMResponse(thinking="Second", code='task_success("second")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        async def async_task() -> str:
            """Async task."""
            pass

        # First run succeeds
        result = await async_task()
        assert result == "first"

        # Set sentinel (directly in KV store)
        state = agent.state()
        _set_cancel_sentinel(state, "async_task")

        # Second run should be cancelled
        with pytest.raises(TaskCancelled):
            await async_task()


class TestCancelledEvent:
    """Tests for CancelledEvent recording."""

    def test_cancelled_event_is_recorded(self):
        """CancelledEvent is added to event log when task is cancelled."""
        from agex import CancelledEvent

        llm = Dummy(
            responses=[
                LLMResponse(thinking="Working", code='task_success("done")'),
                LLMResponse(thinking="Second", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        # First run to initialize state
        my_task()

        # Set sentinel (directly in KV store)
        state = agent.state()
        _set_cancel_sentinel(state, "my_task")

        # Run should raise
        with pytest.raises(TaskCancelled):
            my_task()

        # Check that CancelledEvent was recorded
        state = agent.state()
        event_list = events(state)
        cancelled_events = [e for e in event_list if isinstance(e, CancelledEvent)]

        assert len(cancelled_events) == 1
        assert cancelled_events[0].task_name == "my_task"
        assert cancelled_events[0].iterations_completed == 0

    def test_cancelled_event_callback(self):
        """on_event callback receives CancelledEvent."""
        from agex import CancelledEvent

        llm = Dummy(
            responses=[
                LLMResponse(thinking="Working", code='task_success("done")'),
                LLMResponse(thinking="Second", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        # First run
        my_task()

        # Set sentinel (directly in KV store)
        state = agent.state()
        _set_cancel_sentinel(state, "my_task")

        # Track events via callback
        received_events = []

        with pytest.raises(TaskCancelled):
            my_task(on_event=lambda e: received_events.append(e))

        # Check that callback received CancelledEvent
        cancelled_events = [e for e in received_events if isinstance(e, CancelledEvent)]
        assert len(cancelled_events) == 1
        assert cancelled_events[0].task_name == "my_task"


class TestConcurrentCancellation:
    """Tests for true concurrent cancellation.

    These tests verify that cancel() from one thread/process can interrupt
    a running task in another thread by writing the cancellation sentinel
    directly to the underlying KV store.
    """

    def test_cancel_during_multi_iteration_task(self, tmp_path):
        """Cancel a task mid-execution - verify sentinel is detected between iterations."""

        # Create responses - task_continue indefinitely until cancelled
        llm = Dummy(
            responses=[
                LLMResponse(thinking=f"Iter {i}", code=f'task_continue("{i}")')
                for i in range(20)  # Enough responses for 10 max iterations
            ]
        )
        agent = Agent(
            state=connect_state(type="versioned", storage="disk", path=str(tmp_path)),
            llm=llm,
            name="cancellable_agent",
        )

        @agent.task
        def long_task() -> str:
            """A task that takes multiple iterations."""
            pass

        # Track results and synchronization
        result_holder = {"result": None, "exception": None}
        first_iteration_seen = threading.Event()
        cancel_set = threading.Event()

        def on_event(event):
            # Signal on first iteration
            if hasattr(event, "code") and "task_continue" in str(event.code):
                if not first_iteration_seen.is_set():
                    first_iteration_seen.set()
                # Wait for cancel to be set before proceeding to next iteration
                # This ensures cancel() is called BEFORE the next iteration check
                cancel_set.wait(timeout=5.0)

        def run_task():
            try:
                result_holder["result"] = long_task(on_event=on_event)
            except TaskCancelled as e:
                result_holder["exception"] = e

        # Start task
        task_thread = threading.Thread(target=run_task)
        task_thread.start()

        # Wait for first iteration
        first_iteration_seen.wait(timeout=10.0)
        assert first_iteration_seen.is_set(), "Task didn't complete first iteration"

        # Set cancel - this writes sentinel to KV store
        long_task.cancel()

        # Now let the task proceed - it should detect the cancel on next iteration
        cancel_set.set()

        # Wait for task to complete
        task_thread.join(timeout=10.0)
        assert not task_thread.is_alive(), "Task thread didn't finish"

        # Should have been cancelled
        assert result_holder["exception"] is not None
        assert isinstance(result_holder["exception"], TaskCancelled)
        assert result_holder["exception"].task_name == "long_task"
        # Should have completed at least 1 iteration before cancellation
        assert result_holder["exception"].iterations_completed >= 1

    def test_cancel_before_task_starts(self, tmp_path):
        """Cancel before task even starts - should cancel on first iteration."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="First run", code='task_success("first")'),
                LLMResponse(thinking="Won't run", code='task_success("second")'),
            ]
        )
        agent = Agent(
            state=connect_state(type="versioned", storage="disk", path=str(tmp_path)),
            llm=llm,
        )

        @agent.task
        def my_task() -> str:
            """A task."""
            pass

        # Initialize state by running once
        my_task()

        # Cancel before starting second run
        my_task.cancel()

        # Now run - should be cancelled at iteration 0
        with pytest.raises(TaskCancelled) as exc_info:
            my_task()

        assert exc_info.value.iterations_completed == 0

    def test_cancel_sentinel_persists_to_disk(self, tmp_path):
        """Verify that cancel sentinel is persisted to disk storage."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Init", code='task_success("done")'),
                LLMResponse(thinking="Second", code='task_success("second")'),
            ]
        )
        agent = Agent(
            state=connect_state(type="versioned", storage="disk", path=str(tmp_path)),
            llm=llm,
        )

        @agent.task
        def my_task() -> str:
            """A task."""
            pass

        # Run to initialize state
        my_task()

        # Cancel writes to disk storage
        my_task.cancel()

        # Agent.state() should see the sentinel via same disk storage
        state = agent.state()
        assert _get_cancel_sentinel(state, "my_task") is True

    @pytest.mark.asyncio
    async def test_async_cancel_before_execution(self, tmp_path):
        """Cancel async task before it runs."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Init", code='task_success("first")'),
                LLMResponse(thinking="Won't run", code='task_success("second")'),
            ]
        )
        agent = Agent(
            state=connect_state(type="versioned", storage="disk", path=str(tmp_path)),
            llm=llm,
        )

        @agent.task
        async def async_task() -> str:
            """Async task."""
            pass

        # Initialize
        await async_task()

        # Cancel
        async_task.cancel()

        # Should be cancelled on next run
        with pytest.raises(TaskCancelled):
            await async_task()

    @pytest.mark.asyncio
    async def test_async_cancel_during_multi_iteration_task(self, tmp_path):
        """Cancel an async task mid-execution - verify sentinel is detected between iterations."""
        import asyncio

        # Create responses - task_continue indefinitely until cancelled
        llm = Dummy(
            responses=[
                LLMResponse(thinking=f"Iter {i}", code=f'task_continue("{i}")')
                for i in range(20)  # Enough responses for 10 max iterations
            ]
        )
        agent = Agent(
            state=connect_state(type="versioned", storage="disk", path=str(tmp_path)),
            llm=llm,
            name="cancellable_async_agent",
        )

        @agent.task
        async def long_async_task() -> str:
            """An async task that takes multiple iterations."""
            pass

        # Track results and synchronization
        result_holder = {"result": None, "exception": None}
        first_iteration_seen = asyncio.Event()
        cancel_set = asyncio.Event()

        async def on_event(event):
            # Signal on first iteration
            if hasattr(event, "code") and "task_continue" in str(event.code):
                if not first_iteration_seen.is_set():
                    first_iteration_seen.set()
                # Wait for cancel to be set before proceeding to next iteration
                # This ensures cancel() is called BEFORE the next iteration check
                await asyncio.wait_for(cancel_set.wait(), timeout=5.0)

        async def run_task():
            try:
                result_holder["result"] = await long_async_task(on_event=on_event)
            except TaskCancelled as e:
                result_holder["exception"] = e

        # Start task as background task
        task = asyncio.create_task(run_task())

        # Wait for first iteration
        await asyncio.wait_for(first_iteration_seen.wait(), timeout=10.0)
        assert first_iteration_seen.is_set(), "Task didn't complete first iteration"

        # Set cancel - this writes sentinel to KV store
        long_async_task.cancel()

        # Now let the task proceed - it should detect the cancel on next iteration
        cancel_set.set()

        # Wait for task to complete
        await asyncio.wait_for(task, timeout=10.0)

        # Should have been cancelled
        assert result_holder["exception"] is not None
        assert isinstance(result_holder["exception"], TaskCancelled)
        assert result_holder["exception"].task_name == "long_async_task"
        # Should have completed at least 1 iteration before cancellation
        assert result_holder["exception"].iterations_completed >= 1
