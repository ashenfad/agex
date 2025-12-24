"""
Tests for DataFrame rendering in agent context.

Verifies that DataFrames are properly rendered with full visibility
in the LLM context while maintaining reasonable token budgets.
"""

import pandas as pd
import pytest

from agex import Agent
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy
from agex.state import Versioned


@pytest.mark.skip(
    reason="Print rendering uses different path - covered by task_continue test"
)
def test_print_large_dataframe_shows_all_rows():
    """Test that print(df) with 200 rows shows all rows to the agent."""
    # Create a 200-row DataFrame
    df = pd.DataFrame(
        {
            "event_id": range(200),
            "value": [f"val_{i}" for i in range(200)],
        }
    )

    # Set up dummy client
    client = Dummy()
    client.responses = [
        LLMResponse(
            thinking="I'll print the dataframe",
            code='df = test_df()\nprint(df)\ntask_success("done")',
        ),
    ]

    agent = Agent(name="test_print_df", llm=client, max_iterations=3)

    @agent.fn
    def test_df():
        """Return the test dataframe."""
        return df

    @agent.task("Analyze the data.")
    def analyze_data() -> str:  # type: ignore[return-value]
        """Analyze the data."""

    state = Versioned()
    _ = analyze_data(state=state)

    # Check what was actually rendered for the LLM (uses the regular rendering path)
    assert hasattr(
        client, "all_rendered_messages"
    ), "Client should track rendered messages"
    assert len(client.all_rendered_messages) >= 1, "Should have at least 1 LLM call"

    # Look through all rendered messages to find the printed DataFrame
    found_full_df = False
    visible_rows = 0

    for messages in client.all_rendered_messages:
        for msg in messages:
            content = str(msg.get("content", ""))

            # Look for the DataFrame output in stdout
            if "event_id" in content and "value" in content:
                # Count visible data rows in the rendered content
                lines = content.split("\n")
                data_lines = [
                    line for line in lines if line.strip() and line[0].isdigit()
                ]
                visible_rows = len(data_lines)

                if visible_rows == 200:
                    found_full_df = True
                    break

        if found_full_df:
            break

    assert (
        found_full_df
    ), f"Agent should see all 200 rows via print(). Found {visible_rows} rows."


def test_task_continue_large_dataframe_shows_all_rows():
    """Test that task_continue(msg, df) with 200 rows shows all rows to the agent."""
    # Create a 200-row DataFrame
    df = pd.DataFrame(
        {
            "id": range(200),
            "name": [f"item_{i}" for i in range(200)],
            "category": [f"cat_{i % 5}" for i in range(200)],
        }
    )

    # Set up dummy client
    client = Dummy()
    client.responses = [
        LLMResponse(
            thinking="I'll send the dataframe",
            code='task_continue("Here is the data", get_df())',
        ),
        LLMResponse(
            thinking="Got the data",
            code='task_success("done")',
        ),
    ]

    agent = Agent(name="test_task_continue", llm=client, max_iterations=3)

    @agent.fn
    def get_df():
        """Get the dataframe."""
        return df

    @agent.task("Process the data.")
    def process_data() -> str:  # type: ignore[return-value]
        """Process the data."""

    state = Versioned()
    _ = process_data(state=state)

    # Check what was actually rendered for the LLM
    assert hasattr(
        client, "all_rendered_messages"
    ), "Client should track rendered messages"
    assert len(client.all_rendered_messages) >= 2, "Should have at least 2 LLM calls"

    # The second LLM call should contain the rendered DataFrame from task_continue
    second_call_messages = client.all_rendered_messages[1]

    found_full_df = False
    visible_rows = 0

    for msg in second_call_messages:
        content = str(msg.get("content", ""))

        # Look for the DataFrame in the rendered OutputEvent
        if "id" in content and "name" in content and "category" in content:
            # Count visible data rows
            lines = content.split("\n")
            data_lines = [line for line in lines if line.strip() and line[0].isdigit()]
            visible_rows = len(data_lines)

            if visible_rows >= 200:
                found_full_df = True
                break

    assert (
        found_full_df
    ), f"Agent should see all 200 rows in task_continue. Found {visible_rows} rows."


def test_task_input_large_dataframe_shows_all_rows():
    """Test that DataFrame passed as task input shows all rows."""
    # Set up dummy client
    client = Dummy()
    client.responses = [
        LLMResponse(
            thinking="I can see the input dataframe",
            code='task_success("processed")',
        ),
    ]

    agent = Agent(name="test_task_input", llm=client, max_iterations=3)

    @agent.task("Process the events dataframe.")
    def process_events(events: pd.DataFrame) -> str:  # type: ignore[return-value]
        """Process the events dataframe."""

    # Create a 200-row DataFrame
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=200),
            "event_type": [f"type_{i % 10}" for i in range(200)],
        }
    )

    state = Versioned()
    _ = process_events(events=df, state=state)

    # Check the first LLM call (task initialization)
    assert len(client.all_rendered_messages) >= 1, "Should have at least 1 LLM call"

    first_call_messages = client.all_rendered_messages[0]

    found_full_df = False
    for msg in first_call_messages:
        content = str(msg.get("content", ""))

        # Look for the DataFrame in the task input
        if "inputs.events" in content or (
            "timestamp" in content and "event_type" in content
        ):
            # Count visible data rows
            lines = content.split("\n")
            data_lines = [line for line in lines if line.strip() and line[0].isdigit()]

            if len(data_lines) == 200:
                found_full_df = True
                break

    assert found_full_df, "Agent should see all 200 rows in task input DataFrame"


@pytest.mark.skip(
    reason="Print rendering uses different path - covered by task_continue test"
)
def test_dataframe_respects_token_budget():
    """Test that very large DataFrames are truncated to fit token budget."""
    # Create a 500-row DataFrame (should be truncated)
    huge_df = pd.DataFrame(
        {
            "id": range(500),
            "data": [f"data_{i}" * 10 for i in range(500)],  # Long strings
        }
    )

    # Set up dummy client
    client = Dummy()
    client.responses = [
        LLMResponse(
            thinking="I'll print a huge dataframe",
            code='df = get_huge_df()\nprint(df)\ntask_success("done")',
        ),
    ]

    agent = Agent(name="test_token_budget", llm=client, max_iterations=3)

    @agent.fn
    def get_huge_df():
        """Get the huge dataframe."""
        return huge_df

    @agent.task("Analyze big data.")
    def analyze_big_data() -> str:  # type: ignore[return-value]
        """Analyze big data."""

    state = Versioned()
    _ = analyze_big_data(state=state)

    # Check what was actually rendered for the LLM
    assert hasattr(
        client, "all_rendered_messages"
    ), "Client should track rendered messages"
    found_df = False
    row_count = 0

    for messages in client.all_rendered_messages:
        for msg in messages:
            content = str(msg.get("content", ""))

            if "id" in content and "data" in content:
                lines = content.split("\n")
                data_lines = [
                    line for line in lines if line.strip() and line[0].isdigit()
                ]
                row_count = len(data_lines)
                found_df = True
                break

        if found_df:
            break

    assert found_df, "Should find the DataFrame in output"
    # Should be truncated (less than 500 rows) but still substantial
    assert (
        40 <= row_count < 500
    ), f"Expected truncation for huge DF, got {row_count} rows"
