"""
Efficient event log management using references.

This module provides helpers for adding and retrieving events from the event log
using a reference-based approach that avoids O(N) storage growth.
"""

from typing import Callable

from agex.agent.events import BaseEvent, Event
from agex.state.core import State
from agex.state.versioned import Versioned


def add_event_to_log(
    state: State, event: BaseEvent, on_event: Callable[[BaseEvent], None] | None = None
) -> None:
    """Add an event to the log using references for O(1) storage per event."""
    # If the root state is versioned, stamp the current commit hash on the event
    root_state = state.base_store
    if isinstance(root_state, Versioned) and root_state.current_commit:
        event.commit_hash = root_state.current_commit

    # Set the full_namespace based on the state context
    from agex.state.namespaced import Namespaced

    if isinstance(state, Namespaced):
        # Use the full namespace path from the Namespaced state
        event.full_namespace = state.namespace
    else:
        # For root-level states (Versioned, Live), full_namespace equals agent_name
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
    state.set(event_key, event)

    # Update event log with reference
    event_refs = state.get("__event_log__", [])
    new_refs = event_refs + [event_key]
    state.set("__event_log__", new_refs)


def get_events_from_log(state: State) -> list[Event]:
    """Get events from the state."""
    event_refs = state.get("__event_log__", [])
    # It's possible for events to be added to the log but not yet committed
    # to the state, so we need to handle missing keys gracefully.
    return [state.get(ref) for ref in event_refs if ref in state]
