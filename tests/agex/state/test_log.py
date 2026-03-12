"""Tests for event log management."""

import pytest
from kvgit import store as kvgit_store

from agex.agent.events import ActionEvent, ChapterEvent
from agex.state import _agex_decoder, _agex_encoder
from agex.state.log import (
    add_event_to_log,
    get_events_from_log,
    replace_events_with_chapters,
)


def _make_state():
    return kvgit_store(encoder=_agex_encoder, decoder=_agex_decoder)


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


def test_replace_events_with_chapters():
    """Test replacing a range of events with a chapter."""
    state = _make_state()

    # Add 5 events
    for i in range(5):
        add_event_to_log(
            state,
            ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}"),
        )

    events_before = get_events_from_log(state)
    assert len(events_before) == 5

    # Replace events [1, 3) (indices 1 and 2) with a chapter
    chapter = ChapterEvent(
        agent_name="test",
        name="Middle work",
        message="Did some middle stuff",
    )
    replace_events_with_chapters(state, [(1, 3, chapter)])

    events_after = get_events_from_log(state)
    assert len(events_after) == 4  # 5 - 2 + 1

    # First event unchanged
    assert isinstance(events_after[0], ActionEvent)
    assert events_after[0].thinking == "thought 0"

    # Second is the chapter
    assert isinstance(events_after[1], ChapterEvent)
    assert events_after[1].name == "Middle work"
    assert len(events_after[1].event_refs) == 2

    # Resolve events and verify contents
    resolved = events_after[1].resolve_events(state)
    assert len(resolved) == 2
    assert resolved[0].thinking == "thought 1"
    assert resolved[1].thinking == "thought 2"

    # Remaining events unchanged
    assert isinstance(events_after[2], ActionEvent)
    assert events_after[2].thinking == "thought 3"
    assert isinstance(events_after[3], ActionEvent)
    assert events_after[3].thinking == "thought 4"


def test_replace_events_multiple_chapters():
    """Test replacing multiple non-overlapping ranges."""
    state = _make_state()

    for i in range(6):
        add_event_to_log(
            state,
            ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}"),
        )

    ch1 = ChapterEvent(
        agent_name="test",
        name="First batch",
        message="Events 0-1",
    )
    ch2 = ChapterEvent(
        agent_name="test",
        name="Second batch",
        message="Events 3-4",
    )

    replace_events_with_chapters(state, [(0, 2, ch1), (3, 5, ch2)])

    events_after = get_events_from_log(state)
    assert len(events_after) == 4  # 6 - 2 - 2 + 2

    assert isinstance(events_after[0], ChapterEvent)
    assert events_after[0].name == "First batch"
    assert isinstance(events_after[1], ActionEvent)
    assert events_after[1].thinking == "thought 2"
    assert isinstance(events_after[2], ChapterEvent)
    assert events_after[2].name == "Second batch"
    assert isinstance(events_after[3], ActionEvent)
    assert events_after[3].thinking == "thought 5"


def test_replace_events_overlapping_raises():
    """Test that overlapping ranges raise ValueError."""
    state = _make_state()

    for i in range(5):
        add_event_to_log(
            state,
            ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}"),
        )

    ch1 = ChapterEvent(agent_name="test", name="A", message="a")
    ch2 = ChapterEvent(agent_name="test", name="B", message="b")

    with pytest.raises(ValueError, match="Overlapping"):
        replace_events_with_chapters(state, [(0, 3, ch1), (2, 5, ch2)])


def test_replace_events_out_of_bounds_raises():
    """Test that out-of-bounds ranges raise ValueError."""
    state = _make_state()

    for i in range(3):
        add_event_to_log(
            state,
            ActionEvent(agent_name="test", thinking=f"thought {i}", code=f"x = {i}"),
        )

    ch = ChapterEvent(agent_name="test", name="A", message="a")

    with pytest.raises(ValueError, match="Invalid range"):
        replace_events_with_chapters(state, [(0, 5, ch)])


def test_replace_events_empty_list():
    """Test that empty list is a no-op."""
    state = _make_state()

    add_event_to_log(
        state,
        ActionEvent(agent_name="test", thinking="thought", code="x = 1"),
    )

    replace_events_with_chapters(state, [])

    events = get_events_from_log(state)
    assert len(events) == 1
