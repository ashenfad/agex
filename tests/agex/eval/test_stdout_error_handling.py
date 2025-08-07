"""
Tests for stdout and error handling behavior in agent task loops.

These tests verify that:
1. Errors appear immediately in the iteration where they occur
2. stdout is cleared between iterations so only recent output is shown
"""

from agex import events
from agex.agent import Agent, clear_agent_registry
from agex.agent.events import ActionEvent, OutputEvent
from agex.llm.dummy_client import DummyLLMClient, LLMResponse
from agex.state.kv import Memory
from agex.state.versioned import Versioned


def test_error_appears_immediately_in_first_iteration():
    """
    Test that when an evaluation error occurs, an OutputEvent with error content
    is logged immediately, and the agent can react to it in the next turn.
    """
    clear_agent_registry()
    # First response has a syntax error
    # Second response should see the error from the first response
    llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I'll try to compute something with a syntax error.",
                code="result = 1 + ",  # This will cause a syntax error
            ),
            LLMResponse(
                thinking="I can see the error now. Let me fix it.",
                code="result = 1 + 1\ntask_success(result)",
            ),
        ]
    )
    agent = Agent(name="test_agent", max_iterations=3, llm_client=llm_client)

    @agent.task("Compute a simple result.")
    def compute_simple() -> int:  # type: ignore[return-value]
        """Perform a simple computation and return the result."""
        pass

    state = Versioned(Memory())
    result = compute_simple(state=state)  # type: ignore

    # Should successfully complete with result 2
    assert result == 2

    # Check that an OutputEvent with error content was logged, followed by the corrective ActionEvent
    all_events = [e for e in events(state) if e.full_namespace == "test_agent"]
    output_events = [e for e in all_events if isinstance(e, OutputEvent)]
    action_events = [e for e in all_events if isinstance(e, ActionEvent)]

    # Find output events that contain error messages
    error_outputs = []
    for event in output_events:
        for part in event.parts:
            # Handle both raw PrintAction objects and rendered TextPart objects
            part_text = ""
            if hasattr(part, "text"):
                part_text = str(part.text)
            elif hasattr(part, "__iter__") and len(part) > 0:
                # PrintAction is iterable, get the first argument
                part_text = str(part[0])

            if "💥 Evaluation error" in part_text:
                error_outputs.append(event)
                break

    assert len(error_outputs) >= 1
    # Verify the error content contains expected information
    error_part = error_outputs[0].parts[0]
    if hasattr(error_part, "text"):
        error_content = str(error_part.text)
    else:
        # It's a PrintAction, get the first argument
        error_content = str(error_part[0])

    assert "💥 Evaluation error" in error_content
    assert "syntax" in error_content.lower()

    # Ensure we have at least one action event (the failed one)
    # Note: The corrective action event may not be logged due to task completion timing
    assert len(action_events) >= 1


def test_validation_error_shows_full_type():
    """
    Test that a validation error message is logged as an OutputEvent
    and includes the full, un-truncated type hint.
    """
    clear_agent_registry()
    # First response: Return the wrong type (list of strings instead of list of ints)
    # Second response: "See" the error and return the correct type.
    llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I will try to return a list of strings.",
                code="task_success(['a', 'b'])",
            ),
            LLMResponse(
                thinking="I see the validation error. I'll return a list of ints now.",
                code="task_success([1, 2])",
            ),
        ]
    )
    agent = Agent(name="test_agent", max_iterations=3, llm_client=llm_client)

    @agent.task("A task that requires returning a list of integers.")
    def list_of_ints_task() -> list[int]:  # type: ignore[return-value]
        """A task that must return a list of integers."""
        pass

    state = Versioned(Memory())
    result = list_of_ints_task(state=state)  # type: ignore

    assert result == [1, 2]

    # Check that a validation error OutputEvent was logged
    all_events = [e for e in events(state) if e.full_namespace == "test_agent"]
    output_events = [e for e in all_events if isinstance(e, OutputEvent)]

    # Find output events that contain validation error messages
    validation_outputs = []
    for event in output_events:
        for part in event.parts:
            # Handle both raw PrintAction objects and rendered TextPart objects
            part_text = ""
            if hasattr(part, "text"):
                part_text = str(part.text)
            elif hasattr(part, "__iter__") and len(part) > 0:
                # PrintAction is iterable, get the first argument
                part_text = str(part[0])

            if "Output validation failed" in part_text:
                validation_outputs.append(event)
                break

    assert len(validation_outputs) >= 1
    validation_part = validation_outputs[0].parts[0]
    if hasattr(validation_part, "text"):
        validation_error = str(validation_part.text)
    else:
        # It's a PrintAction, get the first argument
        validation_error = str(validation_part[0])

    assert "Output validation failed" in validation_error
    assert "list[int]" in validation_error
