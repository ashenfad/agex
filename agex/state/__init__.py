"""
A state management system for tic agents.
"""

from ..agent.events import Event
from .core import State, is_live_root
from .kv import KVStore
from .live import Live
from .namespaced import Namespaced
from .scoped import Scoped
from .transient import TransientScope
from .versioned import Versioned

__all__ = [
    "State",
    "is_live_root",
    "Live",
    "KVStore",
    "Namespaced",
    "Scoped",
    "TransientScope",
    "Versioned",
]


def _namespaced(
    state: Versioned | Live | Namespaced, namespaces: list[str]
) -> Namespaced | Versioned | Live:
    if namespaces:
        state = Namespaced(state, namespaces[0])
        if namespaces[1:]:
            return _namespaced(state, namespaces[1:])
    return state


def events(state: Versioned | Live) -> list[Event]:
    """
    Retrieve all events from state.

    Args:
        state: The state object to retrieve events from

    Returns:
        A list of all event objects, sorted chronologically.
        Use full_namespace field to filter by agent paths.

    Examples:
        all_events = events(state)
        worker_a_events = [e for e in all_events if e.full_namespace == "orchestrator/worker_a"]
        orchestrator_tree = [e for e in all_events if e.full_namespace.startswith("orchestrator")]
    """
    # Get root state to traverse all event logs
    root_state = state.base_store

    # Collect events from all event logs in the state
    from agex.state.log import get_events_from_log

    all_events: list[Event] = []

    # Traverse all keys in the root state to find event logs
    for key in root_state.keys():
        if key.endswith("__event_log__"):
            # Extract the namespace path from the key
            if key == "__event_log__":
                # Root-level event log
                log_state = root_state
            else:
                # Namespaced event log - key format is "namespace/path/__event_log__"
                namespace_path = key.replace("/__event_log__", "").split("/")
                log_state = _namespaced(root_state, namespace_path)

            # Get events using the helper that resolves references
            events_list: list[Event] = get_events_from_log(log_state)
            all_events.extend(events_list)

    # Sort events chronologically by timestamp for proper ordering
    all_events.sort(key=lambda event: event.timestamp)

    return all_events
