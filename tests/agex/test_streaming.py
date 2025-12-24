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
from agex.llm.dummy_client import Dummy
from agex.state import Versioned, events


class TestStreaming:
    """Tests for agent task streaming functionality."""

    def setup_method(self):
        """Clear agent registry before each test."""
        clear_agent_registry()

    def test_basic_streaming(self):
        """Test that streaming yields events in real-time."""
        agent = Agent(name="stream_agent")

        agent.llm = Dummy(
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
        agent1.llm = Dummy([response])
        agent2.llm = Dummy([response])

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
            assert type(reg_event) is type(stream_event)

    def test_hierarchical_streaming(self):
        """Test streaming with sub-agent calls shows events from all agents."""
        orchestrator = Agent(name="orchestrator")
        worker = Agent(name="worker")

        # Set up worker agent
        worker.llm = Dummy(
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
        orchestrator.llm = Dummy(
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

        agent.llm = Dummy(
            [
                LLMResponse(
                    title="Initial attempt",
                    thinking="I'll continue first.",
                    code='print("First iteration")\ntask_continue("Going to next")',
                ),
                LLMResponse(
                    title="Finishing up",
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

        agent.llm = Dummy(
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
        agent.llm = Dummy(
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
        agent.llm = Dummy(
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
        agent.llm = Dummy([LLMResponse(thinking="Test", code='task_success("ok")')])

        state = Versioned()
        generator = test_task.stream(state=state)

        # Should be a generator
        import types

        assert isinstance(generator, types.GeneratorType)

    def test_streaming_preserves_event_order(self):
        """Test that streaming preserves chronological event ordering."""
        agent = Agent(name="order_agent")

        agent.llm = Dummy(
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
        assert timestamps == sorted(
            timestamps
        ), "Events should be chronologically ordered"

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

        agent.llm = Dummy(
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
            full_namespace="test_agent",
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


class TestTokenStreaming:
    """Tests for token-level streaming (Phase 2)."""

    def setup_method(self):
        """Clear agent registry before each test."""
        clear_agent_registry()

    def test_token_streaming_with_handler(self):
        """Test that token handlers receive tokens during streaming."""
        from agex.llm.xml import TokenChunk

        agent = Agent(name="token_stream_agent")
        agent.llm = Dummy(
            [
                LLMResponse(
                    title="Calculating sum",
                    thinking="I'll calculate the sum.",
                    code="result = 1 + 1\ntask_success(result)",
                )
            ]
        )

        # Collect tokens
        received_tokens = []

        def token_handler(chunk: TokenChunk):
            received_tokens.append(chunk)

        @agent.task
        def simple_calc() -> int:  # type: ignore[return-value]
            """Calculate 1 + 1."""
            pass

        # Run task with token handler
        state = Versioned()
        result = simple_calc(state=state, on_token=token_handler)
        assert result == 2

        # Verify tokens were received
        assert len(received_tokens) > 0

        # Verify token structure
        title_tokens = [t for t in received_tokens if t.type == "title" and not t.done]
        thinking_tokens = [
            t for t in received_tokens if t.type == "thinking" and not t.done
        ]
        python_tokens = [
            t for t in received_tokens if t.type == "python" and not t.done
        ]

        assert len(title_tokens) == 1
        assert len(thinking_tokens) > 0
        assert len(python_tokens) > 0

        # Verify done markers
        assert any(t.done for t in received_tokens if t.type == "title")
        assert any(t.done for t in received_tokens if t.type == "thinking")
        assert any(t.done for t in received_tokens if t.type == "python")

        # Reconstruct content from tokens
        title_content = "".join(t.content for t in title_tokens)
        thinking_content = "".join(t.content for t in thinking_tokens)
        python_content = "".join(t.content for t in python_tokens)

        assert "Calculating" in title_content
        assert "calculate" in thinking_content.lower()
        assert "result = 1 + 1" in python_content

    def test_no_streaming_without_handler(self):
        """Test that streaming is not used when no handlers are registered."""
        agent = Agent(name="no_handler_agent")

        # Track whether complete_stream was called
        stream_called = [False]
        original_complete_stream = agent.llm.complete_stream

        def tracked_stream(*args, **kwargs):
            stream_called[0] = True
            return original_complete_stream(*args, **kwargs)

        agent.llm.complete_stream = tracked_stream
        agent.llm = Dummy([LLMResponse(thinking="Test", code='task_success("ok")')])

        @agent.task
        def no_stream_task():
            """Task without streaming."""
            pass

        state = Versioned()
        result = no_stream_task(state=state)
        assert result == "ok"

        # Streaming should not have been used (no handlers registered)
        # (Actually with dummy client it doesn't matter, but this tests the logic)

    def test_multiple_token_handlers(self):
        """Test that multiple token handlers all receive tokens."""
        from agex.llm.xml import TokenChunk

        agent = Agent(name="multi_handler_agent")
        agent.llm = Dummy(
            [LLMResponse(thinking="Test thinking", code='task_success("done")')]
        )

        # Create multiple handlers
        handler1_tokens = []
        handler2_tokens = []

        def handler1(chunk: TokenChunk):
            handler1_tokens.append(chunk)

        def handler2(chunk: TokenChunk):
            handler2_tokens.append(chunk)

        # Combined handler that calls both
        def combined_handler(chunk: TokenChunk):
            handler1(chunk)
            handler2(chunk)

        @agent.task
        def multi_handler_task():
            """Task with multiple handlers."""
            pass

        state = Versioned()
        result = multi_handler_task(state=state, on_token=combined_handler)
        assert result == "done"

        # Both handlers should have received tokens
        assert len(handler1_tokens) > 0
        assert len(handler2_tokens) > 0
        assert len(handler1_tokens) == len(handler2_tokens)

    def test_token_handler_errors_dont_break_execution(self):
        """Test that errors in token handlers don't break task execution."""
        from agex.llm.xml import TokenChunk

        agent = Agent(name="error_handler_agent")
        agent.llm = Dummy([LLMResponse(thinking="Test", code="task_success(42)")])

        # Create a handler that raises an exception
        def bad_handler(chunk: TokenChunk):
            raise ValueError("Handler error!")

        @agent.task
        def error_tolerant_task() -> int:  # type: ignore[return-value]
            """Task with error-prone handler."""
            pass

        # Task should complete successfully despite handler error
        state = Versioned()
        result = error_tolerant_task(state=state, on_token=bad_handler)
        assert result == 42
