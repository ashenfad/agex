"""
Efficient event log management using references.

This module provides helpers for adding and retrieving events from the event log
using a reference-based approach that avoids O(N) storage growth.
"""

from typing import Callable

from kvgit import Namespaced, Staged

from agex.agent.events import BaseEvent, ChapterEvent, Event
from agex.state import get_root


def add_event_to_log(
    state, event: BaseEvent, on_event: Callable[[BaseEvent], None] | None = None
) -> None:
    """Add an event to the log using references for O(1) storage per event."""
    # If the root state is versioned, stamp the current commit hash on the event
    root_state = get_root(state)
    if isinstance(root_state, Staged) and root_state.current_commit:
        event.commit_hash = root_state.current_commit

    # Set the full_namespace based on the state context
    if isinstance(state, Namespaced):
        # Use the full namespace path from the Namespaced state
        event.full_namespace = state.namespace
    else:
        # For root-level states (Staged, Live), full_namespace equals agent_name
        event.full_namespace = event.agent_name

    # Call the event handler first, if provided
    if on_event:
        try:
            on_event(event)
        except Exception as e:
            # Log handler error but don't crash the main loop
            print(f"--- Event handler error: {e} ---")

    # Generate unique timestamp-based key
    timestamp_microseconds = int(event.timestamp.timestamp() * 1_000_000)
    event_key = f"_event_{timestamp_microseconds}_"

    # Handle potential timestamp collisions by adding a counter
    counter = 0
    base_key = event_key
    while event_key in state:
        counter += 1
        event_key = f"{base_key}{counter}"

    # Store event separately
    state[event_key] = event

    # Update event log with reference
    event_refs = state.get("__event_log__", [])
    new_refs = event_refs + [event_key]
    state["__event_log__"] = new_refs


def get_events_from_log(state) -> list[Event]:
    """Get events from the state."""
    from agex.agent.datatypes import UnpicklableVariableError

    event_refs = state.get("__event_log__", [])
    # It's possible for events to be added to the log but not yet committed
    # to the state, so we need to handle missing keys gracefully.
    # Also skip events that failed to serialize (e.g. coroutine objects).
    events = []
    for ref in event_refs:
        if ref not in state:
            continue
        try:
            events.append(state.get(ref))
        except UnpicklableVariableError:
            continue
    return events


def replace_events_with_chapters(
    state,
    chapters_and_ranges: list[tuple[int, int, ChapterEvent]],
) -> None:
    """Replace contiguous ranges of events with ChapterEvents.

    Args:
        state: The kvgit state store.
        chapters_and_ranges: List of (start_idx, end_idx, chapter_event) tuples.
            Indices are 0-based into the event refs list. start_idx is inclusive,
            end_idx is exclusive. Ranges must not overlap and must be within bounds.

    Raises:
        ValueError: If ranges overlap or are out of bounds.
    """
    event_refs = state.get("__event_log__", [])
    num_refs = len(event_refs)

    if not chapters_and_ranges:
        return

    # Validate ranges
    sorted_ranges = sorted(chapters_and_ranges, key=lambda x: x[0])
    for i, (start, end, _) in enumerate(sorted_ranges):
        if start < 0 or end > num_refs or start >= end:
            raise ValueError(
                f"Invalid range [{start}, {end}) for event log of length {num_refs}"
            )
        if i > 0:
            prev_end = sorted_ranges[i - 1][1]
            if start < prev_end:
                raise ValueError(
                    f"Overlapping ranges: previous ends at {prev_end}, "
                    f"current starts at {start}"
                )

    # Set commit_hash and full_namespace on chapter events
    root_state = get_root(state)
    for _, _, chapter_event in sorted_ranges:
        if isinstance(root_state, Staged) and root_state.current_commit:
            chapter_event.commit_hash = root_state.current_commit
        if isinstance(state, Namespaced):
            chapter_event.full_namespace = state.namespace
        else:
            chapter_event.full_namespace = chapter_event.agent_name

    # Apply replacements in reverse order to preserve indices
    new_refs = list(event_refs)
    for start, end, chapter_event in reversed(sorted_ranges):
        # Inherit timestamp from the first replaced event so that
        # timestamp-based sorting preserves the correct log position.
        first_ref = event_refs[start]
        if first_ref in state:
            try:
                first_event = state.get(first_ref)
                chapter_event.timestamp = first_event.timestamp
            except Exception:
                pass

        # Store the chapter event with a unique key
        timestamp_microseconds = int(chapter_event.timestamp.timestamp() * 1_000_000)
        event_key = f"_event_{timestamp_microseconds}_"
        counter = 0
        base_key = event_key
        while event_key in state:
            counter += 1
            event_key = f"{base_key}{counter}"

        state[event_key] = chapter_event
        new_refs[start:end] = [event_key]

    state["__event_log__"] = new_refs
