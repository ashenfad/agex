from typing import Any, Iterable

from .core import State


class LiveClosureState(State):
    """
    A read-only, 'live' view into another state, restricted to a set of keys.

    This class is used to represent a function's closure. It doesn't store
    data itself. Instead, it holds a reference to the state where the
    function was defined and the names of the free variables it needs to
    access.

    When a variable is requested, this class performs a live lookup in the
    original state, thus preserving Python's late-binding semantics for
    closures.
    """

    def __init__(self, state_source: State, free_vars: set[str]):
        self._source = state_source
        self._keys = free_vars

    @property
    def base_store(self) -> "State":
        return self._source.base_store

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._keys:
            # This should ideally not be hit if the analyzer is correct.
            raise KeyError(f"'{key}' is not a valid variable in this closure.")
        return self._source.get(key, default)

    def set(self, key: str, value: Any) -> None:
        raise TypeError("Closures are read-only.")

    def remove(self, key: str) -> bool:
        raise TypeError("Closures are read-only.")

    def keys(self) -> Iterable[str]:
        return iter(self._keys)

    def values(self) -> Iterable[Any]:
        for key in self._keys:
            yield self.get(key)

    def items(self) -> Iterable[tuple[str, Any]]:
        for key in self._keys:
            yield key, self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._keys
