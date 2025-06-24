"""
A state wrapper that provides a local scope for temporary operations,
falling back to a parent scope for reads. Writes are always local.
"""

from typing import Any, Iterable

from .core import State
from .ephemeral import Ephemeral


class Scoped(State):
    """
    A state manager that provides a two-tiered scope.

    It checks for keys in a `local_store` first, and if not found, it
    delegates the lookup to a `parent_store`. All writes are confined
    to the `local_store`. This is ideal for managing temporary variables
    in constructs like comprehensions, preventing them from leaking into
    the parent scope.
    """

    def __init__(self, parent_store: State):
        self._local_store = Ephemeral()
        self._parent_store = parent_store
        super().__init__()

    @property
    def base_store(self) -> "State":
        return self._parent_store

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._local_store:
            return self._local_store.get(key, default)
        return self._parent_store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._local_store.set(key, value)

    def remove(self, key: str) -> bool:
        raise NotImplementedError("Not supported for scoped state.")

    def keys(self) -> Iterable[str]:
        raise NotImplementedError("Not supported for scoped state.")

    def values(self) -> Iterable[Any]:
        raise NotImplementedError("Not supported for scoped state.")

    def items(self) -> Iterable[tuple[str, Any]]:
        raise NotImplementedError("Not supported for scoped state.")

    def __contains__(self, key: str) -> bool:
        return key in self._local_store or key in self._parent_store
