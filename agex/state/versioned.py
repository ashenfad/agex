import pickle
import secrets
from dataclasses import dataclass
from typing import Any, Iterable

import xxhash

from . import kv
from .core import State
from .ephemeral import Ephemeral

PARENT_COMMIT = "__parent_commit__%s"
COMMIT_KEYSET = "__commit_keyset__%s"


@dataclass
class SnapshotResult:
    commit_hash: str | None
    unsaved_keys: list[str]


def _get_commit_hash() -> str:
    return secrets.token_hex(8)


def _fast_hash(data: bytes) -> str:
    """Compute fast hash of bytes data."""
    return xxhash.xxh64(data).hexdigest()


class Versioned(State):
    def __init__(self, store: kv.KVStore | None = None, commit_hash: str | None = None):
        if store is None:
            store = kv.Memory()
        self.ephemeral = Ephemeral()
        self.removed = set()
        self.long_term = store
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

    @property
    def base_store(self) -> "State":
        return self

    def _versioned_key(self, key: str, commit_hash: str | None = None) -> str:
        return f"{commit_hash or self.current_commit}:{key}"

    def get(self, key: str, default: Any = None) -> Any:
        # First check ephemeral (in-memory changes)
        if (value := self.ephemeral.get(key)) is not None:
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

                # Track objects for mutation detection only if not already tracked
                # This preserves the original object reference that may have been mutated
                if key not in self.accessed_objects:
                    self.accessed_objects[key] = (original_hash, value)

                return value

        return default

    def set(self, key: str, value: Any) -> None:
        self.ephemeral.set(key, value)
        self.removed.discard(key)
        # Remove from mutation tracking since we're explicitly setting
        self.accessed_objects.pop(key, None)

    def remove(self, key: str) -> bool:
        # Remove from mutation tracking
        self.accessed_objects.pop(key, None)

        if not self.ephemeral.remove(key) and key in self.commit_keys:
            self.removed.add(key)
            return True
        return False

    def keys(self) -> Iterable[str]:
        return set(self.ephemeral.keys()) | set(self.commit_keys.keys()) - self.removed

    def values(self) -> Iterable[Any]:
        for key in self.keys():
            yield self.get(key)

    def items(self) -> Iterable[tuple[str, Any]]:
        for key in self.keys():
            yield key, self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.ephemeral or (
            key not in self.removed and key in self.commit_keys
        )

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

    def _detect_mutations(self) -> dict[str, bytes]:
        """Detect mutations in accessed objects and auto-save them.

        Returns:
            Dict mapping keys to their serialized bytes for mutated objects.
        """
        mutations = {}

        for key, (original_hash, obj_ref) in list(self.accessed_objects.items()):
            # Check ALL accessed objects for mutations, not just unset ones
            # Serialize the object reference we stored
            try:
                current_bytes = pickle.dumps(obj_ref, protocol=pickle.HIGHEST_PROTOCOL)
                current_hash = _fast_hash(current_bytes)
            except Exception:
                # This object was mutated into an unserializable state.
                # We can't get its bytes or hash, but we know it changed.
                # We force it into ephemeral so snapshot() can deal with it.
                if key not in self.ephemeral:
                    self.ephemeral.set(key, obj_ref)
                # It won't be in `mutations`, so snapshot() will try to serialize it.
                continue

            if current_hash != original_hash:
                # Mutation detected! Auto-save it (if not already explicitly set)
                if key not in self.ephemeral:
                    self.ephemeral.set(key, obj_ref)
                # Cache the serialized bytes to avoid re-serializing in snapshot()
                mutations[key] = current_bytes

        return mutations

    def snapshot(self) -> SnapshotResult:
        # First, detect any mutations in accessed objects
        mutations = self._detect_mutations()
        unsaved_keys = []

        if not self.ephemeral:
            # If nothing happened, don't create an empty commit.
            # Just return the current commit hash.
            self.accessed_objects.clear()  # Clear tracking
            return SnapshotResult(self.current_commit, unsaved_keys)

        new_hash = _get_commit_hash()
        diffs = {}
        new_commit_keys = {}

        # Store the order of changes for later diffing.
        diff_keys = tuple(k for k in self.ephemeral.keys() if not k.startswith("__"))
        self.ephemeral.set("__diff_keys__", diff_keys)

        # carry over existing keys that were not removed
        for key, value in self.commit_keys.items():
            if key in self.removed:
                continue
            new_commit_keys[key] = value

        # layer recent writes on top of existing keys
        for key, value in self.ephemeral.items():
            versioned_key = self._versioned_key(key, new_hash)
            # Check if we already have serialized bytes from mutation detection
            serialized_value = None
            if key in mutations:
                serialized_value = mutations[key]
            else:
                # Serialize the value to bytes before storing
                try:
                    serialized_value = pickle.dumps(
                        value, protocol=pickle.HIGHEST_PROTOCOL
                    )
                except Exception:
                    unsaved_keys.append(key)
                    continue

            if serialized_value is not None:
                diffs[versioned_key] = serialized_value
                new_commit_keys[key] = versioned_key

        # Serialize commit metadata
        diffs[COMMIT_KEYSET % new_hash] = pickle.dumps(
            new_commit_keys, protocol=pickle.HIGHEST_PROTOCOL
        )
        diffs[PARENT_COMMIT % new_hash] = pickle.dumps(
            self.current_commit, protocol=pickle.HIGHEST_PROTOCOL
        )

        self.long_term.set_many(**diffs)
        self.commit_keys = new_commit_keys
        self.current_commit = new_hash
        self.removed = set()
        self.ephemeral = Ephemeral()
        self.accessed_objects.clear()  # Clear mutation tracking

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
