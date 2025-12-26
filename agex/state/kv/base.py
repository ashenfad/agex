from abc import ABC, abstractmethod
from typing import Iterable, Mapping


class KVStore(ABC):
    """
    Key-Value store interface that operates on bytes only.

    All values are stored and retrieved as bytes. Serialization/deserialization
    is handled at higher layers (e.g., Versioned state).
    """

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Get bytes value for key, or None if not found."""
        pass

    @abstractmethod
    def set(self, key: str, value: bytes) -> None:
        """Set bytes value for key."""
        pass

    @abstractmethod
    def get_many(self, *args: str) -> Mapping[str, bytes]:
        """Get multiple keys, returning only keys that exist."""
        pass

    @abstractmethod
    def set_many(self, **kwargs: bytes) -> None:
        """Set multiple key-value pairs."""
        pass

    @abstractmethod
    def items(self) -> Iterable[tuple[str, bytes]]:
        """Iterate over all key-value pairs."""
        pass

    @abstractmethod
    def keys(self) -> Iterable[str]:
        """Iterate over all keys."""
        pass

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Check if key exists in store."""
        pass

    @abstractmethod
    def remove(self, key: str) -> None:
        """Remove a key if present."""
        pass

    @abstractmethod
    def remove_many(self, *keys: str) -> None:
        """Remove multiple keys."""
        pass

    @abstractmethod
    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """
        Atomic compare-and-swap operation.

        Set value only if current value equals expected.
        This is required for safe concurrent access to state.

        Args:
            key: The key to update
            value: The new value to set
            expected: The expected current value. None means "must not exist".

        Returns:
            True if swap succeeded (current == expected and value was set)
            False if swap failed (current != expected)

        Example:
            # Only update if value is currently b'old'
            success = store.cas('my_key', b'new', expected=b'old')

            # Create only if key doesn't exist
            success = store.cas('my_key', b'initial', expected=None)
        """
        pass
