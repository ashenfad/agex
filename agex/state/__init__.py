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

from agex.agent.datatypes import UnpicklableMarker, UnpicklableVariableError

from .config import StateConfig
from .live import Live
from .scopes import ScopeSet, scopes

__all__ = [
    # kvgit types (direct)
    "Staged",
    "Live",
    "Namespaced",
    "ConcurrencyError",
    "MergeConflict",
    "MergeResult",
    "KVStore",
    # agex types
    "StateConfig",
    # Utility functions
    "get_root",
    "is_live_root",
    "raw_set",
    "raw_get",
    "raw_remove",
    "commit_state",
    "state_diffs",
    "connect_state",
    "events",
    "scopes",
    "ScopeSet",
]


# ---------------------------------------------------------------------------
# Encoder / decoder with UnpicklableMarker support
# ---------------------------------------------------------------------------
#
# Two flavours of the same logical encoder/decoder pair, picked at module
# import time based on whether numpy is available:
#
# * Plain pickle (1-arg) when numpy isn't installed — same behaviour the
#   module shipped with before the chunked codec existed.
# * kvgit's chunked codec (2-arg, sink/reader) when numpy is importable —
#   numpy ndarrays and pandas DataFrame block buffers are externalized as
#   content-addressed chunks, so a 10 MB DataFrame sliced into N derived
#   variables stores ~10 MB total instead of N×10 MB. Decode semantics
#   match plain pickle (independent, writable copies).
#
# kvgit's ``Staged`` autodetects encoder/decoder arity by signature, so
# every existing call site that passes ``encoder=_agex_encoder,
# decoder=_agex_decoder`` to ``Staged`` keeps working unchanged.
#
# UnpicklableMarker handling is identical in both branches: encode-side
# failures get wrapped in a marker, decode-side failures and marker
# values both raise ``UnpicklableVariableError``.

try:
    # ``scientific`` is kvgit's named codec preset bundling
    # ``NumpyCodec`` (and any future scientific codecs — Arrow,
    # Polars, ... — when kvgit adds them).  Functionally identical to
    # ``compose(NumpyCodec())`` today, but using the preset lets agex
    # pick up additions automatically.  Raises ImportError if numpy
    # isn't available, which is what we want — the except branch
    # below provides a plain-pickle fallback.
    from kvgit.codecs import scientific as _scientific

    _CHUNKED_ENCODER, _CHUNKED_DECODER = _scientific()

    def _agex_encoder(value: Any, sink: Any) -> bytes:
        """Chunked encoder; falls back to UnpicklableMarker on failure."""
        try:
            return _CHUNKED_ENCODER(value, sink)
        except Exception as e:
            marker = UnpicklableMarker(
                variable_name="<unknown>",
                type_name=type(value).__name__,
                original_exception=str(e),
            )
            return pickle.dumps(marker)

    def _agex_decoder(raw: bytes, reader: Any) -> Any:
        """Chunked decoder; raises UnpicklableVariableError on marker / corrupt."""
        try:
            value = _CHUNKED_DECODER(raw, reader)
        except (RecursionError, Exception) as e:
            raise UnpicklableVariableError(
                UnpicklableMarker(
                    variable_name="<unknown>",
                    type_name="<corrupt>",
                    original_exception=f"{type(e).__name__}: {e}",
                )
            )
        if isinstance(value, UnpicklableMarker):
            raise UnpicklableVariableError(value)
        return value

except ImportError:

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
        try:
            value = pickle.loads(raw)
        except (RecursionError, Exception) as e:
            # Catch all deserialization errors: RecursionError, UnpicklingError,
            # EOFError (truncated), AttributeError (missing class), ImportError,
            # ValueError (invalid opcode), TypeError, etc.
            raise UnpicklableVariableError(
                UnpicklableMarker(
                    variable_name="<unknown>",
                    type_name="<corrupt>",
                    original_exception=f"{type(e).__name__}: {e}",
                )
            )
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
# Commit
# ---------------------------------------------------------------------------


def commit_state(
    staged: Staged,
    on_conflict: str = "raise",
) -> MergeResult:
    """Commit staged changes (events, VFS records, file-change markers).

    Suspends filesystem interception during commit so that KV backend I/O
    (e.g., disk writes) doesn't get intercepted by VFS/IsolatedFS patching.

    Args:
        staged: The Staged store to commit.
        on_conflict: Conflict strategy ('raise' or 'abandon').

    Returns:
        MergeResult from kvgit commit.
    """
    from monkeyfs import suspend

    with suspend():
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
        **kwargs: Storage-specific arguments

    Storage-specific kwargs:
        disk:
            path: str - Directory path (required for disk storage)
        indexeddb:
            db_name: str - IndexedDB database name (default: "kvgit")

    Returns:
        A StateConfig instance
    """
    # Validate storage requirements
    if type != "ephemeral" and storage is None:
        raise ValueError(f"State type '{type}' requires storage parameter")

    if storage == "disk" and "path" not in kwargs:
        raise ValueError("Disk storage requires 'path' parameter")

    # Collect optional store-specific parameters
    options = {k: v for k, v in kwargs.items() if k != "path"}

    return StateConfig(
        type=type,
        storage=storage,
        path=kwargs.get("path"),
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
