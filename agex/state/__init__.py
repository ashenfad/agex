"""
A state management system for tic agents.
"""

from ..agent.events import Event
from .core import State, is_ephemeral_root
from .ephemeral import Ephemeral
from .kv import KVStore
from .namespaced import Namespaced
from .scoped import Scoped
from .transient import TransientScope
from .versioned import Versioned

__all__ = [
    "State",
    "is_ephemeral_root",
    "Ephemeral",
    "KVStore",
    "Namespaced",
    "Scoped",
    "TransientScope",
    "Versioned",
]


def _namespaced(
    state: Versioned | Ephemeral | Namespaced, namespaces: list[str]
) -> Namespaced | Versioned | Ephemeral:
    if namespaces:
        state = Namespaced(state, namespaces[0])
        if namespaces[1:]:
            return _namespaced(state, namespaces[1:])
    return state


def events(
    state: Versioned | Ephemeral | Namespaced, *namespaces: str, children: bool = True
) -> list[Event]:
    """
    Retrieve events from state with flexible namespace navigation and hierarchical control.

    Args:
        state: The state object to retrieve events from
        *namespaces: Variable number of namespace path components to navigate to
                    e.g., events(state, "orchestrator", "sub_agent")
        children: Whether to include events from child namespaces (default: True)

    Returns:
        A list of event objects from the specified namespace(s), sorted chronologically.

    Examples:
        events(state)                          # Current namespace + children
        events(state, children=False)          # Current namespace only
        events(state, "agent_name")            # Navigate to agent + children
        events(state, "orchestrator", "sub")   # Navigate to nested path + children
        events(state, "agent", children=False) # Just that agent, no sub-agents
    """
    # Navigate to target namespace if specified
    if namespaces:
        target_state = _namespaced(state, list(namespaces))
    else:
        target_state = state

    # Choose key traversal method based on children parameter and state type
    if children and isinstance(target_state, Namespaced):
        # Use hierarchical traversal for Namespaced when children=True
        keys_to_check = target_state.descendant_keys()
    else:
        # Use direct traversal for all other cases
        keys_to_check = target_state.keys()

    # Collect events from relevant keys
    from agex.state.log import get_events_from_log

    all_events: list[Event] = []
    for key in keys_to_check:
        if key.endswith("__event_log__"):
            # Navigate to the state that contains this event log
            if "/" in key:
                # This is a child namespace event log
                namespace_path = key.replace("/__event_log__", "").split("/")
                log_state = _namespaced(target_state, namespace_path)
            else:
                # This is the current namespace event log
                log_state = target_state

            # Get events using the helper that resolves references
            events_list: list[Event] = get_events_from_log(log_state)
            all_events.extend(events_list)

    # Sort events chronologically by timestamp for proper ordering
    all_events.sort(key=lambda event: event.timestamp)

    return all_events
