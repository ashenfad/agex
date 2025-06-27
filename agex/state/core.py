from abc import ABC, abstractmethod
from typing import Any, Iterable


class State(ABC):
    @property
    @abstractmethod
    def base_store(self) -> "State":
        """Returns the ultimate, non-wrapper state object."""
        pass

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def remove(self, key: str) -> bool:
        pass

    @abstractmethod
    def keys(self) -> Iterable[str]:
        pass

    @abstractmethod
    def values(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def items(self) -> Iterable[tuple[str, Any]]:
        pass

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        pass
