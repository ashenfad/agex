from typing import Any, Iterable, Optional

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

    When pickled (during state snapshots), it automatically freezes itself
    into static storage by capturing the current values of all free variables.
    """

    def __init__(self, state_source: State, free_vars: set[str]):
        self._source: Optional[State] = state_source
        self._keys = free_vars
        self._frozen_store: Optional[dict[str, Any]] = (
            None  # Will be set if we're in frozen state
        )

    @property
    def base_store(self) -> "State":
        if self._frozen_store is not None:
            return self  # We are the base store when frozen
        assert self._source is not None
        return self._source.base_store

    def get(self, key: str, default: Any = None) -> Any:
        # If we're in frozen state, use the frozen store
        if self._frozen_store is not None:
            return self._frozen_store.get(key, default)

        assert self._source is not None

        # Allow access to system variables (starting with __) even if not captured
        if key.startswith("__"):
            return self._source.get(key, default)

        # If the key is in our captured variables, get it from source first
        # This allows user-defined variables to shadow built-ins, matching Python's behavior.
        if key in self._keys:
            return self._source.get(key, default)

        # If not a captured variable, check builtins
        from ..eval.builtins import BUILTINS, STATEFUL_BUILTINS

        if key in BUILTINS:
            return BUILTINS[key]
        if key in STATEFUL_BUILTINS:
            return STATEFUL_BUILTINS[key]

        # If the variable doesn't exist in captured vars or builtins, it's undefined
        from ..eval.error import EvalError

        raise EvalError(f"Name '{key}' is not defined.", None)

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

    def __getstate__(self) -> dict[str, Any]:
        """
        Freeze the live closure into static state when pickling.

        Captures the current values of all free variables and returns them
        as a static dictionary, converting this from a live view to frozen storage.
        """
        from types import ModuleType

        from ..eval.objects import AgexModule

        # If already frozen, return the frozen state
        if self._source is None:
            assert self._frozen_store is not None
            return {"frozen_values": self._frozen_store, "keys": self._keys}

        frozen_values = {}
        for key in self._keys:
            value = self.get(key)  # Use self.get() to properly resolve variables
            # Convert any raw modules to AgexModule (safety net)
            if isinstance(value, ModuleType):
                frozen_values[key] = AgexModule(
                    name=value.__name__, agent_fingerprint=""
                )  # Closure freezing, no agent context
            else:
                frozen_values[key] = value

        return {"frozen_values": frozen_values, "keys": self._keys}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore from frozen state with static storage.

        After unpickling, we behave like static storage rather than a live view.
        """
        self._keys = state["keys"]
        self._source = None  # No longer needed
        self._frozen_store = state["frozen_values"]  # Use frozen values directly
