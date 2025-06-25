import secrets
from typing import Any, Iterable

from . import kv
from .core import State
from .ephemeral import Ephemeral

PARENT_COMMIT = "__parent_commit__%s"
COMMIT_KEYSET = "__commit_keyset__%s"


def _get_commit_hash() -> str:
    return secrets.token_hex(8)


class Versioned(State):
    def __init__(self, store: kv.KVStore, commit_hash: str | None = None):
        self.ephemeral = Ephemeral()
        self.removed = set()
        self.long_term = store
        self.current_commit = commit_hash

        self.commit_keys: dict[str, str]
        if self.current_commit is not None:
            self.commit_keys = self.long_term.get(
                COMMIT_KEYSET % self.current_commit, {}
            )
        else:
            self.commit_keys = {}

    @property
    def base_store(self) -> "State":
        return self

    def _versioned_key(self, key: str, commit_hash: str | None = None) -> str:
        return f"{commit_hash or self.current_commit}:{key}"

    def get(self, key: str, default: Any = None) -> Any:
        if (value := self.ephemeral.get(key)) is not None:
            return value
        if (
            key not in self.removed
            and (versioned_key := self.commit_keys.get(key)) is not None
        ):
            return self.long_term.get(versioned_key, default)
        return default

    def set(self, key: str, value: Any) -> None:
        self.ephemeral.set(key, value)
        self.removed.discard(key)

    def remove(self, key: str) -> bool:
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
            yield current_hash
            current_hash = self.long_term.get(PARENT_COMMIT % current_hash)

    def snapshot(self) -> str | None:
        if not self.ephemeral:
            # If nothing happened, don't create an empty commit.
            # Just return the current commit hash.
            return self.current_commit

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
            diffs[versioned_key] = value
            new_commit_keys[key] = versioned_key

        diffs[COMMIT_KEYSET % new_hash] = new_commit_keys
        diffs[PARENT_COMMIT % new_hash] = self.current_commit

        self.long_term.set_many(**diffs)
        self.commit_keys = new_commit_keys
        self.current_commit = new_hash
        self.removed = set()
        self.ephemeral = Ephemeral()

        return new_hash

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
