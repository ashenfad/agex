"""Test error propagation behavior between parent and child agents."""

import pytest

from agex import Agent
from agex.agent.base import clear_agent_registry
from agex.agent.datatypes import TaskClarify, TaskFail, TaskTimeout
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import DummyLLMClient
from agex.state import Versioned


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the agent registry before and after each test."""
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_top_level_agent_raises_task_clarify():
    """Test that a top-level agent's TaskClarify is raised normally."""
    llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I need more information.",
                code="task_clarify('Please provide more details.')",
            )
        ]
    )
    agent = Agent(
        name="top_level",
        primer="You are a top-level agent.",
        llm_client=llm_client,
    )

    @agent.task
    def my_task() -> str:  # type: ignore[return-value]
        """A task that will request clarification."""
        pass

    # The TaskClarify should be raised normally
    try:
        my_task(state=Versioned())
        assert False, "Expected TaskClarify to be raised"
    except TaskClarify as e:
        assert e.message == "Please provide more details."


def test_sub_agent_converts_task_clarify_to_eval_error():
    """Test that a sub-agent's TaskClarify becomes an EvalError in the parent's stdout."""
    sub_agent_llm = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I need more information.",
                code="task_clarify('Please provide more details.')",
            )
        ]
    )
    sub_agent = Agent(
        name="sub_agent",
        primer="You are a sub-agent.",
        llm_client=sub_agent_llm,
    )

    parent_agent_llm = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I will call the sub-agent and see what happens.",
                code="result = sub_task()",
            )
        ]
    )
    parent_agent = Agent(
        name="parent",
        primer="You are a parent agent.",
        llm_client=parent_agent_llm,
    )

    # Register the sub-agent's task with the parent
    @parent_agent.fn(docstring="Call the sub-agent.")
    @sub_agent.task
    def sub_task() -> str:  # type: ignore[return-value]
        """A task that will request clarification."""
        pass

    # Define the parent's task
    @parent_agent.task
    def parent_task() -> str:  # type: ignore[return-value]
        """Call the sub-agent's task."""
        pass

    # Capture events to verify the parent sees the EvalError
    events_list = []

    def capture_event(event):
        events_list.append(event)

    # The parent should hit max iterations because it can't handle the EvalError
    try:
        parent_task(state=Versioned(), on_event=capture_event)
        assert False, "Expected TaskTimeout to be raised"
    except TaskTimeout:
        # Verify that the parent agent saw the EvalError in an OutputEvent
        output_events = [
            e for e in events_list if hasattr(e, "parts") and e.agent_name == "parent"
        ]
        assert len(output_events) > 0, "Expected at least one OutputEvent from parent"

        # Look for the EvalError message in the output events
        found_eval_error = False
        for event in output_events:
            for part in event.parts:
                if hasattr(part, "__iter__") and len(part) > 0:
                    content = str(part[0])
                    if (
                        "Sub-agent needs clarification: Please provide more details"
                        in content
                    ):
                        found_eval_error = True
                        break
            if found_eval_error:
                break

        assert found_eval_error, (
            f"Expected to find EvalError message in OutputEvents. Events: {[str(e) for e in output_events]}"
        )


def test_top_level_agent_raises_task_fail():
    """Test that a top-level agent's TaskFail is raised normally."""
    llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I cannot complete this task.",
                code="task_fail('Invalid input format.')",
            )
        ]
    )
    agent = Agent(
        name="top_level",
        primer="You are a top-level agent.",
        llm_client=llm_client,
    )

    @agent.task
    def my_task() -> str:  # type: ignore[return-value]
        """A task that will fail."""
        pass

    # The TaskFail should be raised normally
    try:
        my_task(state=Versioned())
        assert False, "Expected TaskFail to be raised"
    except TaskFail as e:
        assert e.message == "Invalid input format."


def test_sub_agent_converts_task_fail_to_eval_error():
    """Test that a sub-agent's TaskFail becomes an EvalError in the parent's stdout."""
    sub_agent_llm = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I cannot complete this task.",
                code="task_fail('Invalid input format.')",
            )
        ]
    )
    sub_agent = Agent(
        name="sub_agent",
        primer="You are a sub-agent.",
        llm_client=sub_agent_llm,
    )

    parent_agent_llm = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I will call the sub-agent and see what happens.",
                code="result = sub_task()",
            )
        ]
    )
    parent_agent = Agent(
        name="parent",
        primer="You are a parent agent.",
        llm_client=parent_agent_llm,
    )

    # Register the sub-agent's task with the parent
    @parent_agent.fn(docstring="Call the sub-agent.")
    @sub_agent.task
    def sub_task() -> str:  # type: ignore[return-value]
        """A task that will fail."""
        pass

    # Define the parent's task
    @parent_agent.task
    def parent_task() -> str:  # type: ignore[return-value]
        """Call the sub-agent's task."""
        pass

    # Capture events to verify the parent sees the EvalError
    events_list = []

    def capture_event(event):
        events_list.append(event)

    # The parent should hit max iterations because it can't handle the EvalError
    try:
        parent_task(state=Versioned(), on_event=capture_event)
        assert False, "Expected TaskTimeout to be raised"
    except TaskTimeout:
        # Verify that the parent agent saw the EvalError in an OutputEvent
        output_events = [
            e for e in events_list if hasattr(e, "parts") and e.agent_name == "parent"
        ]
        assert len(output_events) > 0, "Expected at least one OutputEvent from parent"

        # Look for the EvalError message in the output events
        found_eval_error = False
        for event in output_events:
            for part in event.parts:
                if hasattr(part, "__iter__") and len(part) > 0:
                    content = str(part[0])
                    if "Sub-agent failed: Invalid input format" in content:
                        found_eval_error = True
                        break
            if found_eval_error:
                break

        assert found_eval_error, (
            f"Expected to find EvalError message in OutputEvents. Events: {[str(e) for e in output_events]}"
        )
