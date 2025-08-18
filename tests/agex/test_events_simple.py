"""
Simple, focused tests for the core event system.

Focus on getting basic functionality working before complex scenarios.
"""

from agex import Agent, clear_agent_registry
from agex.agent.events import (
    ActionEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import DummyLLMClient
from agex.state import Versioned, events


class TestEventsSimple:
    """Simple tests for core event functionality."""

    def setup_method(self):
        """Clear agent registry before each test."""
        clear_agent_registry()

    def test_manual_print_creates_output_event(self):
        """Test that manually calling _print_stateful creates OutputEvent."""
        from agex.eval.builtins import _print_stateful
        from agex.state import events

        state = Versioned()
        state.set("__event_log__", [])

        _print_stateful("Test message", state=state, agent_name="test_agent")

        event_list = events(state)
        assert len(event_list) == 1

        event = event_list[0]
        assert isinstance(event, OutputEvent)
        assert event.agent_name == "test_agent"
        assert event.parts == ["Test message"]

    def test_basic_task_events(self):
        """Test that basic agent task execution generates expected events."""
        clear_agent_registry()

        llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll complete this task.",
                    code='task_success("completed")',
                )
            ]
        )
        agent = Agent(name="simple_agent", llm_client=llm_client)

        @agent.task
        def simple_task():
            """Simple task."""
            pass

        state = Versioned()
        result = simple_task(state=state)

        # Get events from the agent's namespace
        event_list = [e for e in events(state) if e.full_namespace == "simple_agent"]

        # Should have TaskStart, Action, and Success events
        assert len(event_list) == 3

        assert isinstance(event_list[0], TaskStartEvent)
        assert event_list[0].agent_name == "simple_agent"
        assert event_list[0].task_name == "simple_task"

        assert isinstance(event_list[1], ActionEvent)
        assert event_list[1].agent_name == "simple_agent"
        assert 'task_success("completed")' in event_list[1].code

        assert isinstance(event_list[2], SuccessEvent)
        assert event_list[2].agent_name == "simple_agent"
        assert event_list[2].result == "completed"

        assert result == "completed"

    def test_simple_print_with_continue(self):
        """Test if print statements work when separated with task_continue."""
        llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll print something.",
                    code='print("Hello")\ntask_continue("Printed")',
                ),
                LLMResponse(thinking="Now I'll finish.", code='task_success("done")'),
            ]
        )
        agent = Agent(name="print_agent", llm_client=llm_client)

        @agent.task
        def print_task():
            """Task that prints."""
            pass

        state = Versioned()
        result = print_task(state=state)

        # Get events from the agent's namespace
        event_list = [e for e in events(state) if e.full_namespace == "print_agent"]

        print("=== EVENTS DEBUG ===")
        for i, event in enumerate(event_list):
            print(f"{i}: {type(event).__name__}")
            if hasattr(event, "parts") and isinstance(event, OutputEvent):
                print(f"   Parts: {event.parts}")

        # Look for any OutputEvents
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]

        # This test will help us understand what's happening
        print(f"Found {len(output_events)} OutputEvents")

        # For now, let's just verify the task completes
        assert result == "done"
        success_events = [e for e in event_list if isinstance(e, SuccessEvent)]
        assert len(success_events) == 1

    def test_trailing_expression_emits_output_event(self):
        """A single trailing expression should auto-print as an OutputEvent."""
        llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll compute then show the value.",
                    code='my_var = "foo"\nmy_var\ntask_success("done")',
                )
            ]
        )
        agent = Agent(name="expr_agent", llm_client=llm_client)

        @agent.task
        def expr_task():
            """Task that ends a block with an expression."""
            pass

        state = Versioned()
        result = expr_task(state=state)

        # Filter events from this agent
        event_list = [e for e in events(state) if e.full_namespace == "expr_agent"]

        # Expect at least one OutputEvent for the trailing expression
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert any(e.parts == ["foo"] for e in output_events)

        # And the task should complete successfully
        assert result == "done"

    def test_trailing_expression_without_task_control_emits_output_event(self):
        """Trailing expression without task_* should still emit an OutputEvent."""
        llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll just evaluate a value.",
                    code='my_var = "foo"\nmy_var',
                )
            ]
        )
        agent = Agent(name="expr_no_end", llm_client=llm_client)

        @agent.task
        def expr_task_no_end():
            """Task that ends with a bare expression and no task control."""
            pass

        state = Versioned()
        try:
            expr_task_no_end(state=state)
        except Exception:
            # Expected (likely TaskTimeout) since there's no task_success/continue
            pass

        event_list = [e for e in events(state) if e.full_namespace == "expr_no_end"]
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]

        # Ensure at least one OutputEvent contains the evaluated value "foo"
        found_foo = any(
            any(isinstance(part, str) and part == "foo" for part in e.parts)
            for e in output_events
        )
        assert found_foo, f"No OutputEvent contained 'foo'. OutputEvents: {[e.parts for e in output_events]}"

    def test_investigate_stateful_builtins(self):
        """Test to investigate how stateful builtins are called during evaluation."""
        llm_client = DummyLLMClient(
            [LLMResponse(thinking="I'll just print.", code='print("Debug print")')]
        )
        agent = Agent(name="investigate_agent", llm_client=llm_client)

        @agent.task
        def investigate_task():
            """Task to investigate builtin calling."""
            pass

        state = Versioned()

        # This will timeout since no task_success, but let's see what events we get
        try:
            investigate_task(state=state)
        except Exception as e:
            print(f"Expected exception: {type(e).__name__}")

        # Get events from the agent's namespace
        event_list = [
            e for e in events(state) if e.full_namespace == "investigate_agent"
        ]

        print("=== INVESTIGATION EVENTS ===")
        for i, event in enumerate(event_list):
            print(f"{i}: {type(event).__name__}")
            if isinstance(event, ActionEvent):
                print(f"   Code: {repr(event.code)}")
            elif isinstance(event, OutputEvent):
                print(f"   Parts: {event.parts}")

        # The investigation shows us what actually happens
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        print(f"OutputEvents found: {len(output_events)}")

        # We expect at least one ActionEvent
        action_events = [e for e in event_list if isinstance(e, ActionEvent)]
        assert len(action_events) >= 1
