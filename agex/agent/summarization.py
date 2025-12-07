"""
Event log summarization for managing long-running agent contexts.

This module provides automatic summarization of older events when the event log
grows beyond configured token limits, similar to garbage collection for memory.
"""

from typing import TYPE_CHECKING

from agex.agent.events import SummaryEvent
from agex.state.core import State
from agex.state.log import get_events_from_log, replace_oldest_events_with_summary

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent


class SummarizationError(Exception):
    """Raised when event log summarization fails."""

    pass


SUMMARIZATION_SYSTEM_MESSAGE = """You are summarizing AI agent execution history.

The agent operates in a Python REPL environment where it:
- Thinks through problems step-by-step
- Writes and executes Python code
- Observes results and adjusts its approach
- Signals completion with task control functions (task_success, task_continue, task_fail, task_clarify)

Your task is to provide a concise summary that captures:
- Key actions the agent took and why
- Important decisions and reasoning
- Outcomes and results
- Any errors or issues encountered

Focus on what the agent accomplished and learned. Be concise but preserve essential context for future actions."""


def maybe_summarize_event_log(agent: "BaseAgent", state: State) -> None:
    """
    Check if event log needs summarization and perform it if necessary.

    Uses high/low water marks to determine when to summarize:
    - If total tokens > high_water: summarize oldest events until < low_water
    - Creates a SummaryEvent via LLM call to replace old events
    - Preserves event storage efficiency (only summary is new)

    Args:
        agent: Agent with llm_client and watermark configuration
        state: State containing the event log

    Raises:
        SummarizationError: If LLM summarization call fails
    """
    # Skip if summarization not configured
    if agent.log_high_water_tokens is None:
        return

    # At this point, log_low_water_tokens is guaranteed to be set
    # (Agent.__init__ ensures it's either explicit or defaulted to 50% of high)
    assert agent.log_low_water_tokens is not None

    # Get current events and compute total tokens
    events = get_events_from_log(state)

    # Need at least 2 events to summarize
    if len(events) < 2:
        return

    total_tokens = sum(event.full_detail_tokens for event in events)

    # Check if we've exceeded high water mark
    if total_tokens <= agent.log_high_water_tokens:
        return

    # First, determine low-detail threshold (75th percentile by age)
    # This allows us to use correct token counts when deciding what to keep
    low_detail_threshold = None
    if len(events) >= 4:  # Need enough events to make it meaningful
        threshold_idx = int(len(events) * 0.75)  # Keep newest 25% at hi-detail
        threshold_event = events[threshold_idx]
        low_detail_threshold = threshold_event.timestamp

    # Determine how many events to summarize
    # Work backwards from newest, keeping events until we're under low_water
    # Use correct token counts: low_detail for old events, full_detail for new events
    events_to_keep = []
    kept_tokens = 0

    for event in reversed(events):
        # Use low_detail_tokens if event is older than threshold
        if low_detail_threshold and event.timestamp < low_detail_threshold:
            event_tokens = event.low_detail_tokens
        else:
            event_tokens = event.full_detail_tokens

        if kept_tokens + event_tokens <= agent.log_low_water_tokens:
            events_to_keep.insert(0, event)
            kept_tokens += event_tokens
        else:
            break

    # Calculate how many to summarize
    num_to_summarize = len(events) - len(events_to_keep)

    # Ensure we're summarizing at least 1 event
    if num_to_summarize < 1:
        # Edge case: even single newest event exceeds low_water
        # Summarize all but the very last event
        num_to_summarize = max(1, len(events) - 1)

    # Get events to summarize
    events_to_summarize = events[:num_to_summarize]
    original_tokens = sum(e.full_detail_tokens for e in events_to_summarize)

    # Call LLM to generate summary (pass events directly for multimodal support)
    try:
        summary_text = agent.llm_client.summarize(
            system=SUMMARIZATION_SYSTEM_MESSAGE,
            content=events_to_summarize,
        )
    except Exception as e:
        raise SummarizationError(
            f"Failed to summarize {num_to_summarize} events: {e}"
        ) from e

    # Create summary event with low-detail threshold
    summary = SummaryEvent(
        agent_name=agent.name,
        summary=summary_text,
        summarized_event_count=num_to_summarize,
        original_tokens=original_tokens,
        low_detail_threshold=low_detail_threshold,
    )

    # Replace old events with summary
    replace_oldest_events_with_summary(state, num_to_summarize, summary)
