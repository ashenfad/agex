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
        """Task raises TaskCancelled when sentinel is set during execution."""
        # Use multiple responses - first iteration sets cancel, second should be cancelled
        llm = Dummy(
            responses=[
                # First iteration - task_continue to allow another iteration
                LLMResponse(thinking="First", code='task_continue("working")'),
                # Second iteration - won't complete due to cancel
                LLMResponse(thinking="Second", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def cancellable_task() -> str:
            """A task that can be cancelled."""
            pass

        iteration_count = [0]

        def on_event(event):
            # Set cancel after first ActionEvent (during first iteration)
            if type(event).__name__ == "ActionEvent":
                iteration_count[0] += 1
                if iteration_count[0] == 1:
                    state = agent.state()
                    _set_cancel_sentinel(state, "cancellable_task")

        # Run should raise TaskCancelled on second iteration
        with pytest.raises(TaskCancelled) as exc_info:
            cancellable_task(on_event=on_event)

        assert exc_info.value.task_name == "cancellable_task"
        assert "cancelled" in exc_info.value.message.lower()

    def test_cancelled_cleans_up_sentinel(self):
        """When task is cancelled, sentinel is removed from state."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="First", code='task_continue("working")'),
                LLMResponse(thinking="Second", code='task_success("ok")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        def on_event(event):
            # Set cancel after first ActionEvent
            if type(event).__name__ == "ActionEvent":
                state = agent.state()
                _set_cancel_sentinel(state, "my_task")

        # Run should raise but clean up sentinel
        with pytest.raises(TaskCancelled):
            my_task(on_event=on_event)

        # Sentinel should be gone (cleaned up by check_cancellation)
        state = agent.state()
        assert _get_cancel_sentinel(state, "my_task") is None

    def test_cancelled_iterations_completed(self):
        """TaskCancelled includes iterations_completed count."""
        # Task that runs multiple iterations before cancel
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Iteration 0", code='task_continue("working")'),
                LLMResponse(thinking="Iteration 1", code='task_continue("working")'),
                LLMResponse(thinking="Won't run", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        iteration_count = [0]

        def on_event(event):
            # Set cancel after second ActionEvent (iteration 1)
            if type(event).__name__ == "ActionEvent":
                iteration_count[0] += 1
                if iteration_count[0] == 2:
                    state = agent.state()
                    _set_cancel_sentinel(state, "my_task")

        # Run should be cancelled at iteration 2
        with pytest.raises(TaskCancelled) as exc_info:
            my_task(on_event=on_event)

        # Should have completed 2 iterations before cancellation
        assert exc_info.value.iterations_completed == 2

    def test_different_tasks_have_separate_sentinels(self):
        """Each task has its own cancel sentinel."""
        llm = Dummy(
            responses=[
                # task_a iter 0 - task_continue then cancel set
                LLMResponse(thinking="A0", code='task_continue("a0")'),
                # task_a iter 1 - not reached, cancel detected
                # task_b uses this next
                LLMResponse(thinking="B", code='task_success("b")'),
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

        def on_event_cancel_a(event):
            # Set cancel for task_a after first ActionEvent
            if type(event).__name__ == "ActionEvent":
                state = agent.state()
                _set_cancel_sentinel(state, "task_a")

        # task_a should be cancelled (cancel set during first iteration)
        with pytest.raises(TaskCancelled):
            task_a(on_event=on_event_cancel_a)

        # task_b should NOT be cancelled (different sentinel)
        result = task_b()
        assert result == "b"

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

    def test_cancelled_tag_rendering(self):
        """Verify that cancellation is rendered in history with the correct XML tag."""
        from agex.llm.xml import TAG_CANCELLED

        llm = Dummy(
            responses=[
                # Run 1: First iteration (task_continue), cancel set during this
                LLMResponse(thinking="Run 1 iter 0", code='task_continue("working")'),
                # Run 1: Won't complete due to cancel
                LLMResponse(thinking="Run 1 iter 1", code='task_success("done")'),
                # Run 2: Should see history of Run 1 cancellation
                LLMResponse(thinking="Run 2", code='task_success("done")'),
            ],
            renderer="xml",  # Enable XML rendering for verification of tags
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        def on_event_set_cancel(event):
            # Set cancel after first ActionEvent
            if type(event).__name__ == "ActionEvent":
                state = agent.state()
                _set_cancel_sentinel(state, "my_task")

        # Run 1: Should raise TaskCancelled and log event
        with pytest.raises(TaskCancelled):
            my_task(on_event=on_event_set_cancel)

        # Run 2: Should see the cancellation in history
        my_task()

        # Inspect messages sent to LLM in the last call
        # all_rendered_messages is list of message lists
        last_call_messages = llm.all_rendered_messages[-1]

        # Convert messages to single string for search
        history_text = str(last_call_messages)

        # Verify XML tag is present
        expected_tag = f"<{TAG_CANCELLED}>"
        assert (
            expected_tag in history_text
        ), f"Expected {expected_tag} in history, got: {history_text}"


class TestTaskCancellationAsync:
    """Async-specific cancellation tests."""

    @pytest.mark.asyncio
    async def test_async_task_raises_cancelled(self):
        """Async task raises TaskCancelled when cancel is set during execution."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="First", code='task_continue("working")'),
                LLMResponse(thinking="Second", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        async def async_task() -> str:
            """Async task."""
            pass

        async def on_event_set_cancel(event):
            # Set cancel after first ActionEvent
            if type(event).__name__ == "ActionEvent":
                state = agent.state()
                _set_cancel_sentinel(state, "async_task")

        # Should be cancelled on second iteration
        with pytest.raises(TaskCancelled):
            await async_task(on_event=on_event_set_cancel)


class TestCancelledEvent:
    """Tests for CancelledEvent recording."""

    def test_cancelled_event_is_recorded(self):
        """CancelledEvent is added to event log when task is cancelled."""
        from agex import CancelledEvent

        llm = Dummy(
            responses=[
                LLMResponse(thinking="First", code='task_continue("working")'),
                LLMResponse(thinking="Second", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        def on_event_set_cancel(event):
            # Set cancel after first ActionEvent
            if type(event).__name__ == "ActionEvent":
                state = agent.state()
                _set_cancel_sentinel(state, "my_task")

        # Run should raise
        with pytest.raises(TaskCancelled):
            my_task(on_event=on_event_set_cancel)

        # Check that CancelledEvent was recorded
        state = agent.state()
        event_list = events(state)
        cancelled_events = [e for e in event_list if isinstance(e, CancelledEvent)]

        assert len(cancelled_events) == 1
        assert cancelled_events[0].task_name == "my_task"
        assert cancelled_events[0].iterations_completed == 1

    def test_cancelled_event_callback(self):
        """on_event callback receives CancelledEvent."""
        from agex import CancelledEvent

        llm = Dummy(
            responses=[
                LLMResponse(thinking="Working", code='task_continue("working")'),
                LLMResponse(thinking="Second", code='task_success("done")'),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def my_task() -> str:
            """Task."""
            pass

        # Track events via callback
        received_events = []
        cancel_set = [False]

        def on_event(event):
            received_events.append(event)
            # Set cancel after first ActionEvent
            if not cancel_set[0] and type(event).__name__ == "ActionEvent":
                cancel_set[0] = True
                state = agent.state()
                _set_cancel_sentinel(state, "my_task")

        with pytest.raises(TaskCancelled):
            my_task(on_event=on_event)

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

    def test_stale_cancel_signal_cleared_at_task_start(self, tmp_path):
        """Stale cancel signal from previous run is cleared when new task starts."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="First run", code='task_success("first")'),
                LLMResponse(
                    thinking="Second run should complete", code='task_success("second")'
                ),
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

        # First run initializes state
        result1 = my_task()
        assert result1 == "first"

        # Set cancel signal (simulating a late cancel after task finished)
        my_task.cancel()

        # Second run should NOT be cancelled - stale signal should be cleared
        result2 = my_task()
        assert result2 == "second"  # Task completed normally, stale signal was cleared

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
    async def test_async_stale_cancel_signal_cleared(self, tmp_path):
        """Stale cancel signal is cleared at async task start."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Init", code='task_success("first")'),
                LLMResponse(
                    thinking="Second run completes", code='task_success("second")'
                ),
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
        result1 = await async_task()
        assert result1 == "first"

        # Set stale cancel (simulating late cancel after task finished)
        async_task.cancel()

        # Should NOT be cancelled - stale signal cleared
        result2 = await async_task()
        assert result2 == "second"

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
