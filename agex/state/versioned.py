from __future__ import annotations

import pickle
import secrets
from dataclasses import dataclass
from typing import Any, Iterable

import xxhash

from ..agent.datatypes import UnpicklableMarker, UnpicklableVariableError
from . import kv
from .core import State
from .live import Live

PARENT_COMMIT = "__parent_commit__%s"
COMMIT_KEYSET = "__commit_keyset__%s"
HEAD_COMMIT = "__head_commit__"
META_KEY = "__meta__%s"
TOTAL_VAR_SIZE_KEY = "__total_var_size__%s"

# key -> (last_touch_counter, size_bytes)
MetaEntry = tuple[int, int | None]


@dataclass
class SnapshotResult:
    commit_hash: str | None
    unsaved_keys: list[str]


def _get_commit_hash() -> str:
    return secrets.token_hex(8)


def get_commit_hash() -> str:
    """
    Public helper to generate a new commit hash.
    """
    return _get_commit_hash()


def _fast_hash(data: bytes) -> str:
    """Compute fast hash of bytes data."""
    return xxhash.xxh64(data).hexdigest()


class Versioned(State):
    def __init__(self, store: kv.KVStore | None = None, commit_hash: str | None = None):
        if store is None:
            store = kv.Memory()
        self.live = Live()
        self.removed = set()
        self.long_term = store

        # If no commit hash provided, try to load HEAD; otherwise create initial commit and set HEAD
        if commit_hash is None:
            head_bytes = store.get(HEAD_COMMIT)
            if head_bytes is not None:
                commit_hash = pickle.loads(head_bytes)
            else:
                commit_hash = _get_commit_hash()
                # Store the initial empty commit metadata so it can be checked out, and set HEAD
                initial_metadata = {
                    COMMIT_KEYSET % commit_hash: pickle.dumps({}),
                    PARENT_COMMIT % commit_hash: pickle.dumps(None),
                    HEAD_COMMIT: pickle.dumps(commit_hash),
                    META_KEY % commit_hash: pickle.dumps({}),
                    TOTAL_VAR_SIZE_KEY % commit_hash: pickle.dumps(0),
                }
                store.set_many(**initial_metadata)

        self.current_commit = commit_hash

        # Track accessed objects for mutation detection
        # key -> (original_hash, object_reference)
        self.accessed_objects: dict[str, tuple[str, Any]] = {}

        self.commit_keys: dict[str, str]
        if self.current_commit is not None:
            commit_keyset_bytes = self.long_term.get(
                COMMIT_KEYSET % self.current_commit
            )
            if commit_keyset_bytes is not None:
                self.commit_keys = pickle.loads(commit_keyset_bytes)
            else:
                self.commit_keys = {}
        else:
            self.commit_keys = {}

        # Metadata tracking for GC/rebase (last touch, serialized size)
        self.meta: dict[str, MetaEntry] = {}
        meta_bytes = self.long_term.get(META_KEY % self.current_commit)
        if meta_bytes is not None:
            try:
                self.meta = pickle.loads(meta_bytes)
            except Exception:
                self.meta = {}
        self._touch_counter = (
            max((entry[0] for entry in self.meta.values()), default=0)
            if self.meta
            else 0
        )

    @property
    def base_store(self) -> "State":
        return self

    @staticmethod
    def _is_user_key(key: str) -> bool:
        return not key.startswith("__")

    def _next_touch(self) -> int:
        self._touch_counter += 1
        return self._touch_counter

    def _note_touch(self, key: str) -> None:
        if not self._is_user_key(key):
            return
        # Always move the touch counter forward for the key
        last_touch = self._next_touch()
        _, size = self.meta.get(key, (last_touch, None))
        self.meta[key] = (last_touch, size)

    def _note_size(self, key: str, size: int) -> None:
        if not self._is_user_key(key):
            return
        last_touch, _ = self.meta.get(key, (self._next_touch(), None))
        self.meta[key] = (last_touch, size)

    def _versioned_key(self, key: str, commit_hash: str | None = None) -> str:
        return f"{commit_hash or self.current_commit}:{key}"

    def get(self, key: str, default: Any = None) -> Any:
        # First check live (in-memory changes)
        if key in self.live:
            value = self.live.get(key)
            self._note_touch(key)
            return value

        # Then check committed state
        if (
            key not in self.removed
            and (versioned_key := self.commit_keys.get(key)) is not None
        ):
            # Get serialized bytes from KV store
            serialized_bytes = self.long_term.get(versioned_key)
            if serialized_bytes is not None:
                # Hash the serialized bytes before deserializing
                original_hash = _fast_hash(serialized_bytes)

                # Deserialize the object
                value = pickle.loads(serialized_bytes)

                # Check if this is an unpicklable marker
                if isinstance(value, UnpicklableMarker):
                    # Strip namespace prefix for display (e.g., "agent_name/cursor" -> "cursor")
                    display_name = key.split("/")[-1] if "/" in key else key
                    raise UnpicklableVariableError(
                        f"Variable '{display_name}' ({value.type_name}) is not available. "
                        f"It was not persisted from a previous execution because "
                        f"it is unpicklable.\n\n"
                        f"Solutions:\n"
                        f"  1. Recreate it: {display_name} = db.cursor()\n"
                        f"  2. Chain operations: results = db.cursor().fetchall()\n"
                        f"  3. Use this variable only within a single turn"
                    )

                # Track objects for mutation detection only if not already tracked
                # This preserves the original object reference that may have been mutated
                if key not in self.accessed_objects:
                    self.accessed_objects[key] = (original_hash, value)

                self._note_touch(key)
                self._note_size(key, len(serialized_bytes))
                return value

        return default

    def set(self, key: str, value: Any) -> None:
        self.live.set(key, value)
        self.removed.discard(key)
        # Remove from mutation tracking since we're explicitly setting
        self.accessed_objects.pop(key, None)
        self._note_touch(key)

    def remove(self, key: str) -> bool:
        # Remove from mutation tracking
        self.accessed_objects.pop(key, None)
        self.meta.pop(key, None)

        removed_from_live = self.live.remove(key)
        removed_from_commit = False
        if not removed_from_live and key in self.commit_keys:
            self.removed.add(key)
            removed_from_commit = True
        return removed_from_live or removed_from_commit

    def keys(self) -> Iterable[str]:
        return set(self.live.keys()) | set(self.commit_keys.keys()) - self.removed

    def values(self) -> Iterable[Any]:
        for key in self.keys():
            yield self.get(key)

    def items(self) -> Iterable[tuple[str, Any]]:
        for key in self.keys():
            yield key, self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.live or (key not in self.removed and key in self.commit_keys)

    def history(self, commit_hash: str | None = None) -> Iterable[str]:
        """
        Return the commit chain given a commit_hash.

        If commit_hash is None, the current commit will be used.
        """
        current_hash = commit_hash or self.current_commit
        while current_hash is not None:
            yield current_hash  # Yield current commit first
            parent_bytes = self.long_term.get(PARENT_COMMIT % current_hash)
            if parent_bytes is not None:
                current_hash = pickle.loads(parent_bytes)
            else:
                current_hash = None

    def _detect_mutations(self) -> tuple[dict[str, bytes], list[str]]:
        """Detect mutations in accessed objects and auto-save them.

        Returns:
            Dict mapping keys to their serialized bytes for mutated objects (or markers).
        """
        mutations = {}
        unsavable_keys = []

        for key, (original_hash, obj_ref) in list(self.accessed_objects.items()):
            # Check ALL accessed objects for mutations, not just unset ones
            # Serialize the object reference we stored
            try:
                current_bytes = pickle.dumps(obj_ref)
                current_hash = _fast_hash(current_bytes)
            except Exception as e:
                # This object was mutated into an unserializable state.
                # Create a marker for it instead.
                if key not in self.live:
                    self.live.set(key, obj_ref)

                # Create marker for the unpicklable mutated object
                marker = UnpicklableMarker(
                    variable_name=key,
                    type_name=type(obj_ref).__name__,
                    original_exception=str(e),
                )
                try:
                    marker_bytes = pickle.dumps(marker)
                    mutations[key] = marker_bytes
                    self._note_touch(key)
                except Exception:
                    # Marker itself failed to pickle (should never happen)
                    unsavable_keys.append(key)
                continue

            if current_hash != original_hash:
                # Mutation detected! Auto-save it (if not already explicitly set)
                if key not in self.live:
                    self.live.set(key, obj_ref)
                # Cache the serialized bytes to avoid re-serializing in snapshot()
                mutations[key] = current_bytes
                self._note_touch(key)

        return mutations, unsavable_keys

    def snapshot(self) -> SnapshotResult:
        # First, detect any mutations in accessed objects
        mutations, unsavable_keys = self._detect_mutations()
        unsaved_keys = list(unsavable_keys)

        if not self.live and not self.removed and not mutations:
            # If nothing changed (no writes, no removals, no mutations), don't create a new commit.
            self.accessed_objects.clear()  # Clear tracking
            return SnapshotResult(self.current_commit, unsaved_keys)

        new_hash = _get_commit_hash()
        diffs = {}
        new_commit_keys = {}
        new_meta: dict[str, MetaEntry] = {}

        # Store the order of changes for later diffing.
        diff_keys = tuple(k for k in self.live.keys() if not k.startswith("__"))
        self.live.set("__diff_keys__", diff_keys)

        # carry over existing keys that were not removed
        for key, value in self.commit_keys.items():
            if key in self.removed:
                continue
            new_commit_keys[key] = value
            if self._is_user_key(key) and key in self.meta:
                new_meta[key] = self.meta[key]

        # layer recent writes on top of existing keys
        for key, value in self.live.items():
            versioned_key = self._versioned_key(key, new_hash)
            # Check if we already have serialized bytes from mutation detection
            serialized_value = None
            if key in mutations:
                serialized_value = mutations[key]
            else:
                # Serialize the value to bytes before storing
                try:
                    serialized_value = pickle.dumps(value)
                except Exception as e:
                    # Value is unpicklable - create a marker instead
                    marker = UnpicklableMarker(
                        variable_name=key,
                        type_name=type(value).__name__,
                        original_exception=str(e),
                    )
                    try:
                        serialized_value = pickle.dumps(marker)
                    except Exception:
                        # Marker itself failed to pickle (should never happen)
                        unsaved_keys.append(key)
                        continue

            if serialized_value is not None:
                diffs[versioned_key] = serialized_value
                new_commit_keys[key] = versioned_key
                if self._is_user_key(key):
                    self._note_size(key, len(serialized_value))
                    if key in self.meta:
                        new_meta[key] = self.meta[key]
                    else:
                        # _note_size will have seeded the meta entry if missing
                        new_meta[key] = self.meta.get(
                            key, (self._touch_counter, len(serialized_value))
                        )

        # Serialize commit metadata and update HEAD
        diffs[COMMIT_KEYSET % new_hash] = pickle.dumps(new_commit_keys)
        diffs[PARENT_COMMIT % new_hash] = pickle.dumps(self.current_commit)
        diffs[HEAD_COMMIT] = pickle.dumps(new_hash)

        # Persist GC/rebase metadata for this commit
        diffs[META_KEY % new_hash] = pickle.dumps(new_meta)
        total_var_size = sum(size for _, size in new_meta.values() if size is not None)
        diffs[TOTAL_VAR_SIZE_KEY % new_hash] = pickle.dumps(total_var_size)

        self.long_term.set_many(**diffs)
        self.commit_keys = new_commit_keys
        self.current_commit = new_hash
        self.removed = set()
        self.live = Live()
        self.accessed_objects.clear()  # Clear mutation tracking
        self.meta = new_meta

        return SnapshotResult(new_hash, unsaved_keys)

    def checkout(self, commit_hash: str) -> "Versioned | None":
        """
        Return a new Versioned state object at a specific commit hash.

        Args:
            commit_hash: The commit to checkout
        """
        # First, validate that the commit is in our history.
        if commit_hash not in list(self.history()):
            return None

        return Versioned(self.long_term, commit_hash=commit_hash)

    def diffs(self, commit_hash: str | None = None) -> dict[str, Any]:
        """
        Returns the state changes for a given commit.

        If commit_hash is None, the current commit will be used.

        Returns:
            An ordered dictionary of state changes.
        """
        target_hash = commit_hash or self.current_commit
        if not target_hash:
            return {}

        commit_state = self.checkout(target_hash)
        if not commit_state:
            # This can happen if the hash is invalid.
            return {}

        # Get ordered state changes
        diff_keys = commit_state.get("__diff_keys__", [])
        return {key: commit_state.get(key) for key in diff_keys}
