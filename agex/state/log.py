"""
Efficient event log management using references.

This module provides helpers for adding and retrieving events from the event log
using a reference-based approach that avoids O(N) storage growth.
"""

from agex.agent.events import BaseEvent, Event
from agex.state.core import State


def add_event_to_log(state: State, event: BaseEvent) -> None:
    """Add an event to the log using references for O(1) storage per event."""
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
    state.set(event_key, event)

    # Update event log with reference
    event_refs = state.get("__event_log__", [])
    new_refs = event_refs + [event_key]
    state.set("__event_log__", new_refs)


def get_events_from_log(state: State) -> list[Event]:
    """Get events from the state."""
    event_refs = state.get("__event_log__", [])
    events = []

    for event_key in event_refs:
        event = state.get(event_key)
        if event:
            events.append(event)

    return events
