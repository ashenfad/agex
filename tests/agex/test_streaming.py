"""
Tests for agent task streaming functionality.

Tests the new streaming capabilities that allow real-time observation
of agent execution through generator-based event streaming.
"""

from agex import Agent, clear_agent_registry
from agex.agent.events import (
    ActionEvent,
    OutputEvent,
    SuccessEvent,
)
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import DummyLLMClient
from agex.state import Versioned, events


class TestStreaming:
    """Tests for agent task streaming functionality."""

    def setup_method(self):
        """Clear agent registry before each test."""
        clear_agent_registry()

    def test_basic_streaming(self):
        """Test that streaming yields events in real-time."""
        agent = Agent(name="stream_agent")

        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll print something then complete the task.",
                    code='print("Hello from agent!")\ntask_success("completed")',
                )
            ]
        )

        @agent.task
        def simple_task():
            """Simple test task."""
            pass

        # Test streaming
        state = Versioned()
        events_list = list(simple_task.stream(state=state))

        # Should have: TaskStart, Action, Output (from print), Success
        event_types = [type(e).__name__ for e in events_list]
        assert "TaskStartEvent" in event_types
        assert "ActionEvent" in event_types
        assert "OutputEvent" in event_types
        assert "SuccessEvent" in event_types

        # Verify chronological order
        assert event_types[0] == "TaskStartEvent"
        assert event_types[-1] == "SuccessEvent"

        # Verify agent names are correct
        for event in events_list:
            assert event.agent_name == "stream_agent"

    def test_streaming_vs_regular_equivalence(self):
        """Test that streaming and regular modes produce identical results."""
        agent1 = Agent(name="regular_agent")
        agent2 = Agent(name="streaming_agent")

        # Same LLM responses for both
        response = LLMResponse(thinking="I'll return 42.", code="task_success(42)")
        agent1.llm_client = DummyLLMClient([response])
        agent2.llm_client = DummyLLMClient([response])

        @agent1.task
        def regular_task() -> int:  # type: ignore[return-value]
            """Return the number 42."""
            pass

        @agent2.task
        def streaming_task() -> int:  # type: ignore[return-value]
            """Return the number 42."""
            pass

        # Test regular mode
        state1 = Versioned()
        result1 = regular_task(state=state1)

        # Test streaming mode
        state2 = Versioned()
        events_list = list(streaming_task.stream(state=state2))

        # Extract result from SuccessEvent
        result2 = None
        for event in events_list:
            if isinstance(event, SuccessEvent):
                result2 = event.result
                break

        # Results should be identical
        assert result1 == result2 == 42

        # Both states should have equivalent events
        regular_events = events(state1)
        streaming_events = events(state2)

        assert len(regular_events) == len(streaming_events)
        for reg_event, stream_event in zip(regular_events, streaming_events):
            assert type(reg_event) == type(stream_event)

    def test_hierarchical_streaming(self):
        """Test streaming with sub-agent calls shows events from all agents."""
        orchestrator = Agent(name="orchestrator")
        worker = Agent(name="worker")

        # Set up worker agent
        worker.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll do some work.",
                    code='print("Worker doing work")\ntask_success("work_done")',
                )
            ]
        )

        @worker.task
        def do_work() -> str:  # type: ignore[return-value]
            """Do some work."""
            pass

        # Register worker task with orchestrator
        orchestrator.fn(do_work)

        # Set up orchestrator agent
        orchestrator.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll delegate to the worker.",
                    code='result = do_work()\nprint(f"Worker returned: {result}")\ntask_success(f"orchestrated: {result}")',
                )
            ]
        )

        @orchestrator.task
        def orchestrate() -> str:  # type: ignore[return-value]
            """Orchestrate work with sub-agents."""
            pass

        # Test streaming captures hierarchical events
        state = Versioned()
        events_list = list(orchestrate.stream(state=state))

        # Verify we see events from both agents
        agent_names = {e.agent_name for e in events_list if hasattr(e, "agent_name")}
        assert "orchestrator" in agent_names
        assert "worker" in agent_names

        # Should have events from both agents
        orchestrator_events = [e for e in events_list if e.agent_name == "orchestrator"]
        worker_events = [e for e in events_list if e.agent_name == "worker"]

        assert len(orchestrator_events) >= 3  # At least TaskStart, Action, Success
        assert len(worker_events) >= 3  # At least TaskStart, Action, Success

        # Verify sub-agent events come as a batch during orchestrator's evaluation
        # (This demonstrates the batching behavior we documented)
        worker_indices = [
            i for i, e in enumerate(events_list) if e.agent_name == "worker"
        ]
        if len(worker_indices) > 1:
            # Worker events should be consecutive (batched)
            for i in range(1, len(worker_indices)):
                assert worker_indices[i] == worker_indices[i - 1] + 1

    def test_streaming_with_multiple_iterations(self):
        """Test streaming with task_continue to verify multiple iterations work."""
        agent = Agent(name="multi_iter_agent")

        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll continue first.",
                    code='print("First iteration")\ntask_continue("Going to next")',
                ),
                LLMResponse(
                    thinking="Now I'll complete.",
                    code='print("Second iteration")\ntask_success("done")',
                ),
            ]
        )

        @agent.task
        def multi_iteration_task():
            """Task with multiple iterations."""
            pass

        state = Versioned()
        events_list = list(multi_iteration_task.stream(state=state))

        # Should have multiple ActionEvents (one per iteration)
        action_events = [e for e in events_list if isinstance(e, ActionEvent)]
        assert len(action_events) == 2

        # Should have multiple OutputEvents (one per print)
        output_events = [e for e in events_list if isinstance(e, OutputEvent)]
        assert len(output_events) >= 2

        # Should end with SuccessEvent
        assert isinstance(events_list[-1], SuccessEvent)
        assert events_list[-1].result == "done"

    def test_streaming_failure_handling(self):
        """Test that streaming properly handles task failures."""
        agent = Agent(name="fail_agent")

        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="This will fail.",
                    code='print("About to fail")\ntask_fail("Something went wrong")',
                )
            ]
        )

        @agent.task
        def failing_task():
            """Task that fails."""
            pass

        state = Versioned()

        # Streaming should yield events even for failed tasks
        events_list = []
        exception_raised = False

        try:
            # Manually iterate to collect events before exception
            for event in failing_task.stream(state=state):
                events_list.append(event)
        except Exception:
            # Failure is expected
            exception_raised = True

        # Should have raised an exception
        assert exception_raised

        # Should have events up to the failure
        event_types = [type(e).__name__ for e in events_list]
        assert "TaskStartEvent" in event_types
        assert "ActionEvent" in event_types
        assert "OutputEvent" in event_types  # From print statement
        assert "FailEvent" in event_types

    def test_streaming_with_persistent_state(self):
        """Test that streaming works correctly with existing state history."""
        agent = Agent(name="persistent_agent")

        # First task execution (regular mode)
        agent.llm_client = DummyLLMClient(
            [LLMResponse(thinking="First task", code='task_success("first")')]
        )

        @agent.task
        def first_task():
            """First task."""
            pass

        state = Versioned()
        result1 = first_task(state=state)
        assert result1 == "first"

        # Second task execution (streaming mode) with same state
        agent.llm_client = DummyLLMClient(
            [LLMResponse(thinking="Second task", code='task_success("second")')]
        )

        @agent.task
        def second_task():
            """Second task."""
            pass

        # Streaming should only show new events, not repeat old ones
        events_list = list(second_task.stream(state=state))

        # Should only have events from second task
        for event in events_list:
            if hasattr(event, "thinking"):
                assert "Second task" in event.thinking
            elif hasattr(event, "result"):
                assert event.result == "second"

    def test_stream_method_exists(self):
        """Test that @agent.task decorated functions have a stream method."""
        agent = Agent(name="method_test_agent")

        @agent.task
        def test_task():
            """Test task."""
            pass

        # Verify stream method exists and is callable
        assert hasattr(test_task, "stream")
        assert callable(test_task.stream)

        # Verify it returns a generator
        agent.llm_client = DummyLLMClient(
            [LLMResponse(thinking="Test", code='task_success("ok")')]
        )

        state = Versioned()
        generator = test_task.stream(state=state)

        # Should be a generator
        import types

        assert isinstance(generator, types.GeneratorType)

    def test_streaming_preserves_event_order(self):
        """Test that streaming preserves chronological event ordering."""
        agent = Agent(name="order_agent")

        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="Multiple outputs",
                    code='print("First")\nprint("Second")\nprint("Third")\ntask_success("done")',
                )
            ]
        )

        @agent.task
        def ordered_task():
            """Task with multiple outputs."""
            pass

        state = Versioned()
        events_list = list(ordered_task.stream(state=state))

        # Verify chronological ordering by timestamp
        timestamps = [e.timestamp for e in events_list]
        assert timestamps == sorted(timestamps), (
            "Events should be chronologically ordered"
        )

        # Verify logical ordering
        event_types = [type(e).__name__ for e in events_list]
        assert event_types[0] == "TaskStartEvent"
        assert event_types[1] == "ActionEvent"
        assert event_types[-1] == "SuccessEvent"

        # OutputEvents should be in sequence
        output_events = [e for e in events_list if isinstance(e, OutputEvent)]
        assert len(output_events) >= 3  # At least 3 print statements

    def test_state_isolation(self):
        """Test that streaming with different states doesn't interfere."""
        agent = Agent(name="isolated_agent")

        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll print something.",
                    code='print("test message")\ntask_success("done")',
                )
            ]
        )

        @agent.task
        def isolated_task():
            """Task for isolation test."""
            pass

        # Run with two separate states
        state1 = Versioned()
        state2 = Versioned()

        events1 = list(isolated_task.stream(state=state1))
        events2 = list(isolated_task.stream(state=state2))

        # Both should have identical event sequences
        assert len(events1) == len(events2)

        # Event types should match
        types1 = [type(e).__name__ for e in events1]
        types2 = [type(e).__name__ for e in events2]
        assert types1 == types2

    def test_output_event_rich_display(self):
        """Test OutputEvent rich display methods."""
        import datetime
        from datetime import timezone

        from agex.agent.events import OutputEvent

        # Create a mock OutputEvent
        event = OutputEvent(
            timestamp=datetime.datetime.now(timezone.utc),
            agent_name="test_agent",
            parts=["simple string", 42, [1, 2, 3]],
        )

        # Test markdown representation (fallback)
        markdown = event._repr_markdown_()
        assert "test_agent" in markdown
        assert "Output:" in markdown
        assert "simple string" in markdown

        # Test HTML representation
        html = event._repr_html_()
        assert "🤖 OutputEvent - test_agent" in html

    def test_output_event_rich_object_handling(self):
        """Test OutputEvent handles rich objects properly."""
        import datetime
        from datetime import timezone

        from agex.agent.events import OutputEvent

        # Mock object with _repr_html_ method
        class MockRichObject:
            def _repr_html_(self):
                return "<div>Rich representation</div>"

        # Mock object with _repr_mimebundle_ method
        class MockMimeBundleObject:
            def _repr_mimebundle_(self, include=None):
                return {"text/html": "<span>Mime bundle HTML</span>"}

        event = OutputEvent(
            timestamp=datetime.datetime.now(timezone.utc),
            agent_name="test_agent",
            parts=[MockRichObject(), MockMimeBundleObject(), "regular string"],
        )

        html = event._repr_html_()

        # Should include rich representations
        assert "Rich representation" in html
        assert "Mime bundle HTML" in html
        assert "regular string" in html

    def test_ipython_formatter_registration(self):
        """Test that IPython formatter registration doesn't crash without IPython."""
        # This test mainly ensures the import error handling works
        # The actual formatter registration can't be easily tested without IPython
        from agex.agent import events

        # Should not raise any exceptions during module import
        # The _register_ipython_formatters() function should handle missing IPython gracefully
        assert events.OutputEvent is not None
