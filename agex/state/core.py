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


def is_live_root(state: State) -> bool:
    """
    Determine if the root state is live (transient) or persistent.

    Follows the base_store chain to the root and checks if it's an Live state.
    This helps determine whether to enforce pickle safety and snapshotting.

    Args:
        state: Any state object (potentially wrapped)

    Returns:
        True if the root state is live, False if it's persistent
    """
    # Import here to avoid circular imports
    from .live import Live

    # Follow base_store chain to the root
    root = state.base_store

    # Check if the root is an Live state
    return isinstance(root, Live)
