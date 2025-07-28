from typing import Any, Iterable

from .core import State
from .live import Live
from .versioned import Versioned


class Namespaced(State):
    def __init__(self, state: "Versioned | Namespaced | Live", namespace: str):
        if "/" in namespace:
            raise ValueError("Namespace names cannot contain '/'")

        # Only allow wrapping base storage states (Versioned, Namespaced, Live)
        # Reject transient evaluation states (Scoped, LiveClosureState, etc.)
        if not isinstance(state, (Versioned, Namespaced, Live)):
            raise TypeError(
                f"Namespaced can only wrap Versioned, Namespaced, or Live states, "
                f"not {type(state).__name__}. Use state.base_store to get the underlying base state."
            )

        self.state = state

        # Build the full namespace path
        if isinstance(state, Namespaced):
            # Wrapping another Namespaced - extend its path
            self.namespace = f"{state.namespace}/{namespace}"
        else:
            # Wrapping Versioned or Live (root level)
            self.namespace = namespace

    @property
    def base_store(self) -> "State":
        return self.state.base_store

    def _local_namespace(self, key: str) -> str | None:
        prefix = f"{self.namespace}/"
        if key.startswith(prefix):
            remainder = key[len(prefix) :]
            # Only return if there are no more slashes (direct child, not nested namespace)
            if remainder and "/" not in remainder:
                return remainder
        return None

    def get(self, key: str, default: Any = None) -> Any:
        return self.base_store.get(f"{self.namespace}/{key}", default)

    def set(self, key: str, value: Any) -> None:
        return self.base_store.set(f"{self.namespace}/{key}", value)

    def remove(self, key: str) -> bool:
        return self.base_store.remove(f"{self.namespace}/{key}")

    def keys(self) -> Iterable[str]:
        return (
            lcl
            for k in self.base_store.keys()
            if (lcl := self._local_namespace(k)) is not None
        )

    def descendant_keys(self) -> Iterable[str]:
        """Get all keys from this namespace and child namespaces (hierarchical traversal)."""
        prefix = f"{self.namespace}/"
        return (
            k[len(prefix) :] for k in self.base_store.keys() if k.startswith(prefix)
        )

    def values(self) -> Iterable[Any]:
        return (self.get(k) for k in self.keys())

    def items(self) -> Iterable[tuple[str, Any]]:
        return ((k, self.get(k)) for k in self.keys())

    def __contains__(self, key: str) -> bool:
        return f"{self.namespace}/{key}" in self.base_store
