"""State management for agex agents.

Uses kvgit types directly:
- Staged — versioned state with commit
- Live — ephemeral in-memory state
- Namespaced — key-prefixed view over any store
"""

import pickle
from typing import Any, Callable, Literal

from kvgit import ConcurrencyError, MergeResult, Namespaced, Staged
from kvgit.errors import MergeConflict
from kvgit.kv import KVStore
from kvgit.versioned import GCVersionedKV as GCVersioned

from agex.agent.datatypes import UnpicklableMarker, UnpicklableVariableError

from .config import StateConfig
from .live import Live

__all__ = [
    # kvgit types (direct)
    "Staged",
    "Live",
    "Namespaced",
    "ConcurrencyError",
    "MergeConflict",
    "MergeResult",
    "GCVersioned",
    "KVStore",
    # agex types
    "StateConfig",
    # Utility functions
    "get_root",
    "is_live_root",
    "raw_set",
    "raw_get",
    "raw_remove",
    "safe_commit",
    "state_diffs",
    "connect_state",
    "events",
]


# ---------------------------------------------------------------------------
# Encoder / decoder with UnpicklableMarker support
# ---------------------------------------------------------------------------


def _agex_encoder(value: Any) -> bytes:
    """Pickle encoder that creates an UnpicklableMarker for unserializable values."""
    try:
        return pickle.dumps(value)
    except Exception as e:
        marker = UnpicklableMarker(
            variable_name="<unknown>",
            type_name=type(value).__name__,
            original_exception=str(e),
        )
        return pickle.dumps(marker)


def _agex_decoder(raw: bytes) -> Any:
    """Pickle decoder that raises UnpicklableVariableError on marker values."""
    value = pickle.loads(raw)
    if isinstance(value, UnpicklableMarker):
        raise UnpicklableVariableError(value)
    return value


# ---------------------------------------------------------------------------
# Root navigation (replaces base_store chain)
# ---------------------------------------------------------------------------


def get_root(state: Any) -> Any:
    """Get the root store from any state wrapper.

    Navigates through Namespaced wrappers to reach the underlying
    root store (Staged or Live).
    """
    if isinstance(state, Namespaced):
        return state._store  # kvgit flattens nested namespaces
    return state  # Staged, Live, or unknown


def is_live_root(state: Any) -> bool:
    """Check if the root state is Live (ephemeral/transient)."""
    return isinstance(get_root(state), Live)


# ---------------------------------------------------------------------------
# Raw KV access (replaces set_raw / get_raw / remove_raw)
# ---------------------------------------------------------------------------


def raw_set(staged: Staged, key: str, value: Any) -> None:
    """Write directly to KV store, bypassing versioning.

    Used for cross-process signals like cancellation.
    """
    staged.versioned.store.set(key, pickle.dumps(value))


def raw_get(staged: Staged, key: str) -> Any | None:
    """Read directly from KV store, bypassing versioning.

    Returns None if key doesn't exist.
    """
    raw = staged.versioned.store.get(key)
    return pickle.loads(raw) if raw else None


def raw_remove(staged: Staged, key: str) -> None:
    """Remove directly from KV store, bypassing versioning."""
    staged.versioned.store.remove(key)


# ---------------------------------------------------------------------------
# Commit with mutation detection (replaces snapshot + merge)
# ---------------------------------------------------------------------------


def safe_commit(
    staged: Staged,
    referenced_keys: set[str] | None = None,
    on_conflict: str = "raise",
) -> MergeResult:
    """Commit staged changes with optional mutation detection.

    Suspends filesystem interception during commit so that KV backend I/O
    (e.g., disk writes) doesn't get intercepted by VFS/IsolatedFS patching.

    Args:
        staged: The Staged store to commit.
        referenced_keys: State keys referenced in agent code. Keys present
            in state but not explicitly staged are re-staged so the encoder
            runs and kvgit detects byte-level changes from in-place mutations.
        on_conflict: Conflict strategy ('raise' or 'abandon').

    Returns:
        MergeResult from kvgit commit.
    """
    from monkeyfs import suspend

    with suspend():
        if referenced_keys:
            state_keys = staged.keys()
            for key in referenced_keys & set(state_keys):
                if not staged.is_staged(key):
                    # Re-stage so encoder runs and kvgit detects byte changes.
                    # Skip keys that are already UnpicklableMarkers — they can't
                    # have been mutated in-place since agent code can't access them.
                    try:
                        staged[key] = staged.get(key)
                    except UnpicklableVariableError:
                        pass

        return staged.commit(on_conflict=on_conflict)


# ---------------------------------------------------------------------------
# State diffs (replaces Versioned.diffs)
# ---------------------------------------------------------------------------


def state_diffs(staged: Staged, commit_hash: str | None = None) -> dict[str, Any]:
    """Get state changes for a specific commit.

    Uses kvgit's native diff() to compare the commit against its parent
    and returns the key-value pairs that were added or modified.
    """
    target = commit_hash or staged.current_commit
    if not target:
        return {}

    versioned = staged.versioned
    parents = versioned.parents(target)

    if parents:
        diff_result = versioned.diff(parents[0], target)
        changed_keys = diff_result.added | diff_result.modified
    else:
        # First commit — everything is new
        view = staged.checkout(target)
        if not view:
            return {}
        changed_keys = {k for k in view.keys() if not k.startswith("__")}

    # Read values from a checkout at the target commit
    view = staged.checkout(target)
    if not view:
        return {}
    return {key: view.get(key) for key in changed_keys if not key.startswith("__")}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def connect_state(
    type: Literal["ephemeral", "versioned", "live"],
    storage: str | None = None,
    init: "Callable[[], dict[str, Any]] | dict[str, Any] | None" = None,
    **kwargs,
) -> StateConfig:
    """Create a state configuration.

    Args:
        type: State semantics ("ephemeral", "versioned", or "live")
        storage: Storage backend ("memory", "disk", or "indexeddb").
            Not required for ephemeral.
        init: Callable or dict to initialize state variables on first session creation.
        **kwargs: Type and storage-specific arguments

    Storage-specific kwargs:
        disk:
            path: str - Directory path (required for disk storage)
        indexeddb:
            db_name: str - IndexedDB database name (default: "kvgit")

    Type-specific kwargs (versioned):
        high_water_bytes: int - Trigger GC when total size exceeds this
        low_water_bytes: int - Target size after GC (default: 80% of high_water)

    Returns:
        A StateConfig instance
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


# ---------------------------------------------------------------------------
# Namespacing helper
# ---------------------------------------------------------------------------


def _namespaced(state: Any, namespaces: list[str]) -> Any:
    """Wrap a state with one or more namespace levels."""
    result = state
    for ns in namespaces:
        result = Namespaced(result, ns)
    return result


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def events(state: Any) -> list:
    """Retrieve all events from state.

    Args:
        state: The state object to retrieve events from

    Returns:
        A list of all event objects, sorted chronologically.
    """
    from agex.agent.events import Event
    from agex.state.log import get_events_from_log

    root_state = get_root(state)

    all_events: list[Event] = []

    # Traverse all keys in the root state to find event logs
    for key in root_state.keys():
        if key.endswith("__event_log__"):
            if key == "__event_log__":
                log_state = root_state
            else:
                namespace_path = key.replace("/__event_log__", "").split("/")
                log_state = _namespaced(root_state, namespace_path)

            events_list: list[Event] = get_events_from_log(log_state)
            all_events.extend(events_list)

    all_events.sort(key=lambda event: event.timestamp)

    return all_events
