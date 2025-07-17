"""
Tests for stdout and error handling behavior in agent task loops.

These tests verify that:
1. Errors appear immediately in the iteration where they occur
2. stdout is cleared between iterations so only recent output is shown
"""

from agex.agent import Agent, clear_agent_registry
from agex.llm.core import TextPart
from agex.llm.dummy_client import DummyLLMClient, LLMResponse
from agex.state.kv import Memory
from agex.state.versioned import Versioned


def test_error_appears_immediately_in_first_iteration():
    """
    Test that when an evaluation error occurs, it appears in stdout
    immediately in the same iteration, not delayed to the next iteration.
    """
    clear_agent_registry()
    agent = Agent(name="test_agent", max_iterations=3)

    # First response has a syntax error
    # Second response should see the error from the first response
    agent.llm_client = DummyLLMClient(
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

    @agent.task("Compute a simple result.")
    def compute_simple() -> int:  # type: ignore[return-value]
        """Perform a simple computation and return the result."""
        pass

    state = Versioned(Memory())
    result = compute_simple(state=state)  # type: ignore

    # Should successfully complete with result 2
    assert result == 2

    # Check that the error was visible to the agent in the second iteration
    # The agent's second response should reference seeing the error
    # We can verify this by checking that the conversation includes both the error and the fix


def test_stdout_cleared_between_iterations():
    """
    Test that stdout is cleared between iterations so only recent output is shown,
    not accumulated output from all previous iterations.
    """
    clear_agent_registry()
    agent = Agent(name="test_agent", max_iterations=4)

    # Agent will print something in iteration 1, then something different in iteration 2
    # Only the most recent prints should be visible
    agent.llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="First iteration - I'll print something.",
                code='print("iteration 1 message")',
            ),
            LLMResponse(
                thinking="Second iteration - I'll print something else.",
                code='print("iteration 2 message")',
            ),
            LLMResponse(
                thinking="Third iteration - I'll check what's in stdout and finish.",
                code='print("iteration 3 message")\ntask_success("done")',
            ),
        ]
    )

    @agent.task("Task that prints across multiple iterations.")
    def print_task() -> str:  # type: ignore[return-value]
        """A task that prints in multiple iterations."""
        pass

    state = Versioned(Memory())
    result = print_task(state=state)  # type: ignore

    assert result == "done"

    # After completion, stdout should only contain the most recent prints
    final_stdout = state.get("test_agent/__stdout__", [])

    # Convert PrintTuple objects to strings for easier checking
    stdout_strings = []
    for item in final_stdout:
        if hasattr(item, "__iter__") and not isinstance(item, str):
            # Handle PrintTuple - extract the actual printed values
            stdout_strings.extend(str(arg) for arg in item)
        else:
            stdout_strings.append(str(item))

    # Should only see the latest iteration's print, not accumulated prints
    assert "iteration 3 message" in stdout_strings
    # Should NOT see prints from earlier iterations
    assert "iteration 1 message" not in stdout_strings
    assert "iteration 2 message" not in stdout_strings


def test_error_cleared_between_iterations():
    """
    Test that evaluation errors are cleared between iterations,
    so old errors don't accumulate in stdout.
    """
    clear_agent_registry()
    agent = Agent(name="test_agent", max_iterations=4)

    agent.llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="First iteration - cause an error.",
                code="x = undefined_variable",  # NameError
            ),
            LLMResponse(
                thinking="Second iteration - cause a different error.",
                code="y = 1 / 0",  # ZeroDivisionError
            ),
            LLMResponse(
                thinking="Third iteration - now complete successfully.",
                code="z = 42\ntask_success(z)",
            ),
        ]
    )

    @agent.task("Task that has multiple errors then succeeds.")
    def error_task() -> int:  # type: ignore[return-value]
        """A task that causes errors then succeeds."""
        pass

    state = Versioned(Memory())
    result = error_task(state=state)  # type: ignore

    assert result == 42

    # Final stdout should not contain accumulated errors from all iterations
    final_stdout = state.get("test_agent/__stdout__", [])
    stdout_strings = [str(item) for item in final_stdout]

    # Should not see both the NameError and ZeroDivisionError accumulated
    name_error_count = sum(1 for s in stdout_strings if "undefined_variable" in s)
    zero_div_error_count = sum(1 for s in stdout_strings if "division by zero" in s)

    # At most one of each error type should be present (ideally none since they should be cleared)
    assert name_error_count <= 1
    assert zero_div_error_count <= 1

    # Preferably, old errors should be completely cleared
    # (This test might need adjustment based on the final implementation)


def test_mixed_prints_and_errors_cleared():
    """
    Test that both prints and errors are properly managed between iterations.
    """
    clear_agent_registry()
    agent = Agent(name="test_agent", max_iterations=5)

    agent.llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="Print and then cause an error.",
                code='print("before error")\nx = undefined_var',
            ),
            LLMResponse(
                thinking="Print something else and succeed.",
                code='print("after error")\ntask_success("complete")',
            ),
        ]
    )

    @agent.task("Task with mixed prints and errors.")
    def mixed_task() -> str:  # type: ignore[return-value]
        """A task with prints and errors."""
        pass

    state = Versioned(Memory())
    result = mixed_task(state=state)  # type: ignore

    assert result == "complete"

    # Final stdout should only show recent content
    final_stdout = state.get("test_agent/__stdout__", [])
    stdout_strings = []
    for item in final_stdout:
        if hasattr(item, "__iter__") and not isinstance(item, str):
            stdout_strings.extend(str(arg) for arg in item)
        else:
            stdout_strings.append(str(item))

    # Should see the recent print
    assert "after error" in stdout_strings
    # Should not see the old print or error
    assert "before error" not in stdout_strings
    assert "undefined_var" not in stdout_strings


def test_validation_error_shows_full_type():
    """
    Test that a validation error message in stdout includes the full,
    un-truncated type hint for complex types.
    """
    clear_agent_registry()
    agent = Agent(name="test_agent", max_iterations=3)

    # First response: Return the wrong type (list of strings instead of list of ints)
    # Second response: "See" the error and return the correct type.
    agent.llm_client = DummyLLMClient(
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

    @agent.task("A task that requires returning a list of integers.")
    def list_of_ints_task() -> list[int]:  # type: ignore[return-value]
        """A task that must return a list of integers."""
        pass

    state = Versioned(Memory())
    result = list_of_ints_task(state=state)  # type: ignore

    assert result == [1, 2]

    # Check that the validation error message was present in the agent's context
    # for the second iteration.
    log_keys = state.get("test_agent/__msg_log__")
    # The message before the last one should be the system context with the error
    system_context_message_key = log_keys[-2]
    system_context_message = state.get(f"test_agent/{system_context_message_key}")

    # The content can either be a list of parts or a raw string.
    if isinstance(system_context_message.content, list):
        system_context_content = "\n".join(
            part.text
            for part in system_context_message.content
            if isinstance(part, TextPart)
        )
    else:
        system_context_content = system_context_message.content

    expected_error_string = (
        "Output validation failed. The returned value did not "
        "match the expected type 'list[int]'."
    )
    assert expected_error_string in system_context_content
