from typing import Any, Iterator

from .core import State


class Namespaced(State):
    def __init__(self, state: State, namespace: str):
        if "/" in namespace:
            raise ValueError("Namespace names cannot contain '/'")
        self.state = state
        self.namespace = namespace

    @property
    def base_store(self) -> "State":
        return self.state.base_store

    def _local_namespace(self, key: str) -> str | None:
        path = key.split("/")
        if len(path) > 1 and path[-2] == self.namespace:
            return path[-1]
        return None

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(f"{self.namespace}/{key}", default)

    def set(self, key: str, value: Any) -> None:
        return self.state.set(f"{self.namespace}/{key}", value)

    def remove(self, key: str) -> bool:
        return self.state.remove(f"{self.namespace}/{key}")

    def keys(self) -> Iterator[str]:
        return (
            lcl
            for k in self.base_store.keys()
            if (lcl := self._local_namespace(k)) is not None
        )

    def values(self) -> Iterator[Any]:
        return (self.get(k) for k in self.keys())

    def items(self) -> Iterator[tuple[str, Any]]:
        return ((k, self.get(k)) for k in self.keys())

    def __contains__(self, key: str) -> bool:
        return f"{self.namespace}/{key}" in self.state
