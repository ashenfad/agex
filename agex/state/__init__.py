"""A state management system for tic agents."""

from typing import Any, Callable, Literal, cast

from ..agent.events import Event
from .config import StateConfig
from .core import State, is_live_root
from .gc import GCVersioned, RebaseResult
from .kv import KVStore
from .live import Live
from .namespaced import Namespaced
from .scoped import Scoped
from .versioned import ConcurrencyError, Versioned, get_commit_hash

__all__ = [
    "State",
    "StateConfig",
    "is_live_root",
    "get_commit_hash",
    "Live",
    "KVStore",
    "Namespaced",
    "Scoped",
    "Versioned",
    "ConcurrencyError",
    "RebaseResult",
    "GCVersioned",
    "connect_state",
]


def connect_state(
    type: Literal["ephemeral", "versioned", "live"],
    storage: str | None = None,
    init: "Callable[[], dict[str, Any]] | dict[str, Any] | None" = None,
    **kwargs,
) -> StateConfig:
    """
    Create a state configuration.

    Args:
        type: State semantics ("ephemeral", "versioned", or "live")
        storage: Storage backend ("memory" or "disk"). Not required for ephemeral.
        init: Callable or dict to initialize state variables on first session creation.
              If a callable, it will be invoked when the session is first created.
              The returned dict keys become variable names in the agent's namespace.
        **kwargs: Type and storage-specific arguments

    Storage-specific kwargs:
        disk:
            path: str - Directory path (required for disk storage)

    Type-specific kwargs (versioned):
        high_water_bytes: int - Trigger GC when total size exceeds this
        low_water_bytes: int - Target size after GC (default: 80% of high_water)

    Returns:
        A StateConfig instance

    Examples:
        # Ephemeral (no persistence)
        connect_state(type="ephemeral")

        # In-memory versioned (for testing)
        connect_state(type="versioned", storage="memory")

        # Disk-backed versioned with GC
        connect_state(
            type="versioned",
            storage="disk",
            path="~/.agex/state",
            high_water_bytes=100_000_000,
        )

        # Versioned with initial variables
        connect_state(
            type="versioned",
            storage="disk",
            path="/tmp/agex/tmnt",
            init=lambda: {"leo": load_cal("leo.ics"), ...},
        )
    """
    # Validate storage requirements
    if type != "ephemeral" and storage is None:
        raise ValueError(f"State type '{type}' requires storage parameter")

    if storage == "disk" and "path" not in kwargs:
        raise ValueError("Disk storage requires 'path' parameter")

    # Validate GC params only apply to versioned state
    gc_params = [k for k in ("high_water_bytes", "low_water_bytes") if k in kwargs]
    if gc_params and type != "versioned":
        raise ValueError(
            f"GC parameters ({', '.join(gc_params)}) only apply to "
            f"'versioned' state, but got type='{type}'"
        )

    # Collect optional store-specific parameters
    options = {
        k: v
        for k, v in kwargs.items()
        if k not in ("path", "high_water_bytes", "low_water_bytes")
    }

    return StateConfig(
        type=type,
        storage=storage,
        path=kwargs.get("path"),
        high_water_bytes=kwargs.get("high_water_bytes"),
        low_water_bytes=kwargs.get("low_water_bytes"),
        options=options if options else None,
        init=init,
    )


def _namespaced(state: State, namespaces: list[str]) -> State:
    base = cast(Versioned | Namespaced | Live, state)
    if namespaces:
        base = Namespaced(base, namespaces[0])
        if namespaces[1:]:
            return _namespaced(base, namespaces[1:])
    return base


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
