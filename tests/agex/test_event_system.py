"""
Comprehensive test suite for the event system.

Tests all event types with dummy LLM responses to ensure proper event creation,
attribution, and filtering.
"""

import pytest

from agex import Agent, clear_agent_registry
from agex.agent.events import (
    ActionEvent,
    ErrorEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import DummyLLMClient
from agex.state import Versioned, events
from agex.state.log import add_event_to_log


class TestEventSystem:
    """Test suite for validating the complete event system."""

    def setup_method(self):
        """Clear agent registry before each test."""
        clear_agent_registry()

    def test_task_start_event_creation(self):
        """Test that TaskStartEvent is properly created with all required fields."""
        agent = Agent(name="test_agent")

        @agent.task
        def test_task(name: str, count: int = 5):
            """Test task with parameters."""
            pass

        # Set up dummy LLM to complete the task
        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll complete this simple task.",
                    code='task_success("completed")',
                )
            ]
        )

        state = Versioned()
        result = test_task("test_value", count=10, state=state)

        # Get events from the agent's namespace
        event_list = events(state, "test_agent", children=False)
        task_start_events = [e for e in event_list if isinstance(e, TaskStartEvent)]

        assert len(task_start_events) == 1
        event = task_start_events[0]

        # Validate all fields
        assert event.agent_name == "test_agent"
        assert event.task_name == "test_task"
        assert event.inputs == {"name": "test_value", "count": 10}
        assert isinstance(event.message, str)
        assert len(event.message) > 0
        assert "Test task with parameters." in event.message
        assert "test_value" in event.message
        assert "task_success" in event.message

    def test_action_event_creation(self):
        """Test that ActionEvent captures agent thinking and code."""
        agent = Agent(name="thinking_agent")

        @agent.task
        def think_task():
            """Task that requires thinking."""
            pass

        thinking_text = "I need to analyze this problem step by step."
        code_text = 'result = "analyzed"\ntask_success(result)'

        agent.llm_client = DummyLLMClient(
            [LLMResponse(thinking=thinking_text, code=code_text)]
        )

        state = Versioned()
        result = think_task(state=state)

        # Get events from the agent's namespace
        event_list = events(state, "thinking_agent", children=False)
        action_events = [e for e in event_list if isinstance(e, ActionEvent)]

        assert len(action_events) == 1
        event = action_events[0]

        assert event.agent_name == "thinking_agent"
        assert event.thinking == thinking_text
        assert event.code == code_text

    def test_output_event_creation(self):
        """Test that OutputEvent is created for print(), help(), dir() calls."""
        agent = Agent(name="output_agent")

        @agent.task
        def output_task():
            """Task that produces various outputs."""
            pass

        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll test various output functions and complete.",
                    code='print("Hello World")\nhelp()\ndir()\ntask_success("done")',
                )
            ]
        )

        state = Versioned()
        result = output_task(state=state)

        # Get events from the agent's namespace
        event_list = events(state, "output_agent", children=False)
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]

        # Should have events for print(), help(), and dir()
        assert len(output_events) >= 3

        # All should have correct agent name
        for event in output_events:
            assert event.agent_name == "output_agent"
            assert isinstance(event.parts, list)
            assert len(event.parts) > 0

    def test_success_event_creation(self):
        """Test that SuccessEvent is created when task_success() is called."""
        agent = Agent(name="success_agent")

        @agent.task
        def success_task():
            """Task that succeeds with a result."""
            pass

        expected_result = {"status": "completed", "value": 42}
        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll return a successful result.",
                    code=f"task_success({expected_result})",
                )
            ]
        )

        state = Versioned()
        result = success_task(state=state)

        # Get events from the agent's namespace
        event_list = events(state, "success_agent", children=False)
        success_events = [e for e in event_list if isinstance(e, SuccessEvent)]

        assert len(success_events) == 1
        event = success_events[0]

        assert event.agent_name == "success_agent"
        assert event.result == expected_result
        assert result == expected_result

    def test_fail_event_creation(self):
        """Test that FailEvent is created when task_fail() is called."""
        agent = Agent(name="fail_agent")

        @agent.task
        def fail_task():
            """Task that fails."""
            pass

        fail_message = "This task cannot be completed"
        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I cannot complete this task.",
                    code=f'task_fail("{fail_message}")',
                )
            ]
        )

        state = Versioned()

        # Task should raise TaskFail exception
        with pytest.raises(Exception) as exc_info:
            fail_task(state=state)

        # Get events from the agent's namespace
        event_list = events(state, "fail_agent", children=False)
        fail_events = [e for e in event_list if isinstance(e, FailEvent)]

        assert len(fail_events) == 1
        event = fail_events[0]

        assert event.agent_name == "fail_agent"
        assert event.message == fail_message

    def test_multi_agent_event_attribution(self):
        """Test that events are properly attributed in multi-agent scenarios."""
        clear_agent_registry()

        # Create multiple agents
        agent1 = Agent(name="agent_one")
        agent2 = Agent(name="agent_two")
        orchestrator = Agent(name="orchestrator")

        # Create dual-decorated functions
        @orchestrator.fn(docstring="Function handled by agent one")
        @agent1.task("Complete task one")
        def task_one(value: int):
            pass

        @orchestrator.fn(docstring="Function handled by agent two")
        @agent2.task("Complete task two")
        def task_two(value: int):
            pass

        @orchestrator.task("Orchestrate both tasks")
        def orchestrate(input_value: int):
            pass

        # Set up dummy responses
        agent1.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="Agent one processing",
                    code="result = inputs.value * 2\ntask_success(result)",
                )
            ]
        )

        agent2.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="Agent two processing",
                    code="result = inputs.value + 10\ntask_success(result)",
                )
            ]
        )

        orchestrator.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll call both sub-agents and return results",
                    code="r1 = task_one(inputs.input_value)\nr2 = task_two(inputs.input_value)\ntask_success({'r1': r1, 'r2': r2})",
                )
            ]
        )

        shared_state = Versioned()
        result = orchestrate(input_value=5, state=shared_state)

        # Verify the result is correct
        assert result == {"r1": 10, "r2": 15}

        # Check orchestrator events in its namespaced state
        orchestrator_events = events(shared_state, "orchestrator", children=False)
        orchestrator_agent_names = {
            e.agent_name for e in orchestrator_events if hasattr(e, "agent_name")
        }
        assert "orchestrator" in orchestrator_agent_names

        # Orchestrator should have TaskStart, Action, and Success events
        has_task_start = any(isinstance(e, TaskStartEvent) for e in orchestrator_events)
        has_action = any(isinstance(e, ActionEvent) for e in orchestrator_events)
        has_success = any(isinstance(e, SuccessEvent) for e in orchestrator_events)

        assert has_task_start, "Orchestrator missing TaskStartEvent"
        assert has_action, "Orchestrator missing ActionEvent"
        assert has_success, "Orchestrator missing SuccessEvent"

        # Check agent1 events in its namespaced state
        agent1_events = events(
            shared_state, "orchestrator", "agent_one", children=False
        )
        agent1_agent_names = {
            e.agent_name for e in agent1_events if hasattr(e, "agent_name")
        }
        assert "agent_one" in agent1_agent_names

        # Agent1 should have TaskStart, Action, and Success events
        has_task_start = any(isinstance(e, TaskStartEvent) for e in agent1_events)
        has_action = any(isinstance(e, ActionEvent) for e in agent1_events)
        has_success = any(isinstance(e, SuccessEvent) for e in agent1_events)

        assert has_task_start, "Agent1 missing TaskStartEvent"
        assert has_action, "Agent1 missing ActionEvent"
        assert has_success, "Agent1 missing SuccessEvent"

        # Check agent2 events in its namespaced state
        agent2_events = events(
            shared_state, "orchestrator", "agent_two", children=False
        )
        agent2_agent_names = {
            e.agent_name for e in agent2_events if hasattr(e, "agent_name")
        }
        assert "agent_two" in agent2_agent_names

        # Agent2 should have TaskStart, Action, and Success events
        has_task_start = any(isinstance(e, TaskStartEvent) for e in agent2_events)
        has_action = any(isinstance(e, ActionEvent) for e in agent2_events)
        has_success = any(isinstance(e, SuccessEvent) for e in agent2_events)

        assert has_task_start, "Agent2 missing TaskStartEvent"
        assert has_action, "Agent2 missing ActionEvent"
        assert has_success, "Agent2 missing SuccessEvent"

    def test_event_filtering_excludes_error_events(self):
        """Test that ErrorEvents are filtered out of agent conversation."""
        from agex.agent.conversation import conversation_log

        agent = Agent(name="filter_test_agent")
        state = Versioned()

        # Manually add events including an ErrorEvent
        add_event_to_log(
            state,
            TaskStartEvent(
                agent_name="filter_test_agent",
                task_name="test",
                inputs={},
                message="Test task",
            ),
        )
        add_event_to_log(
            state,
            ActionEvent(
                agent_name="filter_test_agent",
                thinking="Thinking...",
                code="print('hello')",
            ),
        )
        add_event_to_log(
            state,
            ErrorEvent(
                agent_name="filter_test_agent",
                error=ValueError("This is a framework error"),
                recoverable=True,
            ),
        )
        add_event_to_log(
            state, SuccessEvent(agent_name="filter_test_agent", result="completed")
        )

        # Generate conversation log (should filter out ErrorEvent)
        messages = conversation_log(state, "System message", agent)

        # Should have: system message, initial task message, action message
        # Should NOT include anything from ErrorEvent
        assert len(messages) >= 2

        # Check that no message contains error content
        message_content = " ".join(str(msg.content) for msg in messages)
        assert "framework error" not in message_content
        assert "ValueError" not in message_content

    def test_complete_task_lifecycle_events(self):
        """Test that a complete task generates all expected events in correct order."""
        agent = Agent(name="lifecycle_agent")

        @agent.task
        def lifecycle_task(input_data: str):
            """Complete lifecycle test task."""
            pass

        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll process this input and return a result.",
                    code='print(f"Processing: {inputs.input_data}")\nresult = inputs.input_data.upper()\nprint(f"Result: {result}")\ntask_success(result)',
                )
            ]
        )

        state = Versioned()
        result = lifecycle_task("hello world", state=state)

        # Get events from the agent's namespace
        event_list = events(state, "lifecycle_agent", children=False)

        # Verify event sequence and types
        event_types = [type(event).__name__ for event in event_list]

        # Should start with TaskStartEvent
        assert event_types[0] == "TaskStartEvent"

        # Should have ActionEvent
        assert "ActionEvent" in event_types

        # Should have OutputEvents from print statements
        assert "OutputEvent" in event_types
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert len(output_events) >= 2  # Two print statements

        # Should end with SuccessEvent
        assert event_types[-1] == "SuccessEvent"

        # Verify all events have correct agent name
        for event in event_list:
            if hasattr(event, "agent_name"):
                assert event.agent_name == "lifecycle_agent"

        # Verify final result
        assert result == "HELLO WORLD"

    def test_event_persistence_in_versioned_state(self):
        """Test that events are properly persisted and can be retrieved after state snapshots."""
        agent = Agent(name="persistence_agent")

        @agent.task
        def persistence_task():
            """Test event persistence."""
            pass

        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="Testing persistence", code='task_success("persisted")'
                )
            ]
        )

        state = Versioned()
        result = persistence_task(state=state)

        # Take a snapshot
        snapshot_result = state.snapshot()
        assert len(snapshot_result.unsaved_keys) == 0  # All should be saved

        # Get events from the agent's namespace
        event_list = events(state, "persistence_agent", children=False)

        # Should have persistent events
        assert len(event_list) > 0

        # Verify events survived persistence
        has_task_start = any(isinstance(e, TaskStartEvent) for e in event_list)
        has_success = any(isinstance(e, SuccessEvent) for e in event_list)

        assert has_task_start
        assert has_success
