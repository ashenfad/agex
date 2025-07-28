from typing import Any, Iterable

from agex.state.core import State


class Live(State):
    """
    In-memory state that is not persisted. Good for multi-step workflows
    that need to work with unpickleable objects like database connections.
    """

    def __init__(self):
        self.store = {}

    @property
    def base_store(self) -> "State":
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value

    def remove(self, key: str) -> bool:
        if key in self.store:
            del self.store[key]
            return True
        return False

    def keys(self) -> Iterable[str]:
        return self.store.keys()

    def values(self) -> Iterable[Any]:
        return self.store.values()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self.store.items()

    def __contains__(self, key: str) -> bool:
        return key in self.store

    def __len__(self) -> int:
        return len(self.store)

    def __bool__(self) -> bool:
        return bool(self.store)
