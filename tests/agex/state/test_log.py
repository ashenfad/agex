"""Tests for event log management."""

import pytest
from kvit import store as kvit_store

from agex.agent.events import ActionEvent, SummaryEvent
from agex.state import _agex_decoder, _agex_encoder
from agex.state.log import (
    add_event_to_log,
    get_events_from_log,
    replace_oldest_events_with_summary,
)


def _make_state():
    return kvit_store(encoder=_agex_encoder, decoder=_agex_decoder)


def test_add_and_get_events():
    """Test basic event log operations."""
    state = _make_state()

    # Add some events
    event1 = ActionEvent(agent_name="test", thinking="thought 1", code="x = 1")
    event2 = ActionEvent(agent_name="test", thinking="thought 2", code="x = 2")

    add_event_to_log(state, event1)
    add_event_to_log(state, event2)

    events = get_events_from_log(state)
    assert len(events) == 2
    assert events[0].thinking == "thought 1"
    assert events[1].thinking == "thought 2"


def test_replace_oldest_events_with_summary():
    """Test replacing oldest events with a summary."""
    state = _make_state()

    # Add 5 events
    for i in range(5):
        event = ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}")
        add_event_to_log(state, event)

    # Verify initial state
    events = get_events_from_log(state)
    assert len(events) == 5
    assert all(isinstance(e, ActionEvent) for e in events)

    # Create summary for first 3 events
    summary = SummaryEvent(
        agent_name="test",
        summary="Summary of first 3 events",
        summarized_event_count=3,
        original_tokens=100,
    )

    # Replace oldest 3 events
    replace_oldest_events_with_summary(state, 3, summary)

    # Verify result
    events = get_events_from_log(state)
    assert len(events) == 3  # 1 summary + 2 kept events
    assert isinstance(events[0], SummaryEvent)
    assert events[0].summary == "Summary of first 3 events"
    assert events[0].summarized_event_count == 3

    # Verify kept events are the newer ones (indices 3 and 4 from original)
    assert isinstance(events[1], ActionEvent)
    assert events[1].thinking == "thought 3"
    assert isinstance(events[2], ActionEvent)
    assert events[2].thinking == "thought 4"


def test_replace_oldest_validates_count():
    """Test that replace_oldest_events_with_summary validates count parameter."""
    state = _make_state()

    # Add 3 events
    for i in range(3):
        event = ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}")
        add_event_to_log(state, event)

    summary = SummaryEvent(
        agent_name="test",
        summary="Summary",
        summarized_event_count=2,
        original_tokens=50,
    )

    # Test count <= 0
    with pytest.raises(ValueError, match="count must be > 0"):
        replace_oldest_events_with_summary(state, 0, summary)

    with pytest.raises(ValueError, match="count must be > 0"):
        replace_oldest_events_with_summary(state, -1, summary)

    # Test count > log length
    with pytest.raises(
        ValueError, match="Cannot replace 5 events, log only has 3 events"
    ):
        replace_oldest_events_with_summary(state, 5, summary)


def test_replace_oldest_edge_cases():
    """Test edge cases for replace_oldest_events_with_summary."""
    state = _make_state()

    # Add 3 events
    for i in range(3):
        event = ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}")
        add_event_to_log(state, event)

    # Replace exactly 1 event
    summary1 = SummaryEvent(
        agent_name="test",
        summary="Summary of 1 event",
        summarized_event_count=1,
        original_tokens=30,
    )
    replace_oldest_events_with_summary(state, 1, summary1)

    events = get_events_from_log(state)
    assert len(events) == 3  # 1 summary + 2 kept
    assert isinstance(events[0], SummaryEvent)
    assert events[1].thinking == "thought 1"
    assert events[2].thinking == "thought 2"

    # Replace all but one (2 events)
    summary2 = SummaryEvent(
        agent_name="test",
        summary="Summary of 2 events",
        summarized_event_count=2,
        original_tokens=60,
    )
    replace_oldest_events_with_summary(state, 2, summary2)

    events = get_events_from_log(state)
    assert len(events) == 2  # 1 summary + 1 kept
    assert isinstance(events[0], SummaryEvent)
    assert events[0].summary == "Summary of 2 events"
    assert events[1].thinking == "thought 2"


def test_replace_preserves_event_storage():
    """Test that replacing events reuses existing event storage (no duplication)."""
    state = _make_state()

    # Add events
    for i in range(5):
        event = ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}")
        add_event_to_log(state, event)

    # Get all keys before replacement
    event_refs_before = state.get("__event_log__", [])
    kept_refs = event_refs_before[3:]  # Events we expect to keep

    # Replace first 3
    summary = SummaryEvent(
        agent_name="test",
        summary="Summary",
        summarized_event_count=3,
        original_tokens=90,
    )
    replace_oldest_events_with_summary(state, 3, summary)

    # Get refs after
    event_refs_after = state.get("__event_log__", [])

    # Verify that the kept event refs are the same (no duplication)
    assert event_refs_after[1:] == kept_refs

    # Verify old events are still in storage (just not referenced by log)
    for old_ref in event_refs_before[:3]:
        assert old_ref in state  # Old events still exist in storage


def test_cascading_summaries():
    """Test that summaries can themselves be summarized."""
    state = _make_state()

    # Add 10 events
    for i in range(10):
        event = ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}")
        add_event_to_log(state, event)

    # First summarization: replace first 5
    summary1 = SummaryEvent(
        agent_name="test",
        summary="Summary of events 0-4",
        summarized_event_count=5,
        original_tokens=200,
    )
    replace_oldest_events_with_summary(state, 5, summary1)

    events = get_events_from_log(state)
    assert len(events) == 6  # 1 summary + 5 kept

    # Second summarization: replace first 3 (which includes the summary)
    summary2 = SummaryEvent(
        agent_name="test",
        summary="Summary of summary + 2 more events",
        summarized_event_count=3,
        original_tokens=150,  # Original tokens from what we're replacing
    )
    replace_oldest_events_with_summary(state, 3, summary2)

    events = get_events_from_log(state)
    assert len(events) == 4  # 1 new summary + 3 kept
    assert isinstance(events[0], SummaryEvent)
    assert events[0].summary == "Summary of summary + 2 more events"
