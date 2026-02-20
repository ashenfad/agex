"""
A state wrapper that provides a local scope for temporary operations,
falling back to a parent scope for reads. Writes are always local.
"""

from typing import Any, Iterable

from kvit import Live


class Scoped:
    """
    A state manager that provides a two-tiered scope.

    It checks for keys in a `local_store` first, and if not found, it
    delegates the lookup to a `parent_store`. All writes are confined
    to the `local_store`. This is ideal for managing temporary variables
    in constructs like comprehensions, preventing them from leaking into
    the parent scope.
    """

    def __init__(self, parent_store):
        self._local_store = Live()
        self._parent_store = parent_store

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._local_store:
            return self._local_store.get(key, default)
        return self._parent_store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        # For closure variables, write through to the underlying state
        # This allows UserFunction callbacks to modify captured variables
        from .closure import LiveClosureState

        if isinstance(self._parent_store, LiveClosureState):
            if (
                key in self._parent_store._keys
                and self._parent_store._source is not None
            ):
                self._parent_store._source.set(key, value)
                return
        self._local_store.set(key, value)

    def remove(self, key: str) -> bool:
        # Only remove from local scope, don't delegate to parent
        # This matches Python's scoping: del only affects current scope
        if key in self._local_store:
            self._local_store.remove(key)
            return True
        return False

    def keys(self) -> Iterable[str]:
        local = self._local_store.keys()
        outer = self._parent_store.keys()
        return set(local) | set(outer)

    def values(self) -> Iterable[Any]:
        raise NotImplementedError("Not supported for scoped state.")

    def items(self) -> Iterable[tuple[str, Any]]:
        raise NotImplementedError("Not supported for scoped state.")

    def get_many(self, *keys: str) -> dict[str, Any]:
        return {k: self.get(k) for k in keys if k in self}

    def __contains__(self, key: str) -> bool:
        return key in self._local_store or key in self._parent_store

    def __getitem__(self, key: str) -> Any:
        if key in self:
            return self.get(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        self.remove(key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(set(self.keys()))
