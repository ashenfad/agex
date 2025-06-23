import copy
import secrets
from typing import Any, Iterator

from . import kv
from .closure import LiveClosureState
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

    def keys(self) -> Iterator[str]:
        ks = set(self.ephemeral.keys()) | set(self.commit_keys.keys()) - self.removed
        return iter(ks)

    def values(self) -> Iterator[Any]:
        for key in self.keys():
            yield self.get(key)

    def items(self) -> Iterator[tuple[str, Any]]:
        for key in self.keys():
            yield key, self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.ephemeral or (
            key not in self.removed and key in self.commit_keys
        )

    def history(self, commit_hash: str | None = None) -> Iterator[str]:
        """
        Return the commit chain given a commit_hash.

        If commit_hash is None, the current commit will be used.
        """
        current_hash = commit_hash or self.current_commit
        while current_hash is not None:
            yield current_hash
            current_hash = self.long_term.get(PARENT_COMMIT % current_hash)

    def snapshot(self) -> str:
        new_hash = _get_commit_hash()
        diffs = {}
        new_commit_keys = {}

        # carry over existing keys that were not removed
        for key, value in self.commit_keys.items():
            if key in self.removed:
                continue
            new_commit_keys[key] = value

        # layer recent writes on top of existing keys
        for key, value in self.ephemeral.items():
            # This is the "freeze" logic for closures.
            if hasattr(value, "closure_state") and isinstance(
                value.closure_state, LiveClosureState
            ):
                # Resolve the live closure into a static, ephemeral one.
                frozen_closure = Ephemeral()
                for k, v in value.closure_state.items():
                    frozen_closure.set(k, v)
                # Replace the closure on a shallow copy of the function object
                value = copy.copy(value)
                value.closure_state = frozen_closure

            versioned_key = self._versioned_key(key, new_hash)
            diffs[versioned_key] = value
            new_commit_keys[key] = versioned_key

        diffs[COMMIT_KEYSET % new_hash] = new_commit_keys
        diffs[PARENT_COMMIT % new_hash] = self.current_commit

        self.long_term.set_many(**diffs)
        self.commit_keys = new_commit_keys
        self.current_commit = new_hash
        self.removed = set()

        return new_hash
