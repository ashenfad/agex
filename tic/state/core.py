from abc import ABC, abstractmethod
from typing import Any, Iterator


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
    def keys(self) -> Iterator[str]:
        pass

    @abstractmethod
    def values(self) -> Iterator[Any]:
        pass

    @abstractmethod
    def items(self) -> Iterator[tuple[str, Any]]:
        pass

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        pass
