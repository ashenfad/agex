from abc import ABC, abstractmethod
from typing import Iterable, Mapping


class KVStore(ABC):
    """
    Key-Value store interface that operates on bytes only.

    All values are stored and retrieved as bytes. Serialization/deserialization
    is handled at higher layers (e.g., Versioned state).
    """

    @abstractmethod
    def get(self, key: str, default: bytes | None = None) -> bytes | None:
        """Get bytes value for key, or default if not found."""
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
    def __contains__(self, key: str) -> bool:
        """Check if key exists in store."""
        pass


class Memory(KVStore):
    """A memory-backed KV store that stores values as bytes."""

    def __init__(self):
        self.memory: dict[str, bytes] = {}

    def get(self, key: str, default: bytes | None = None) -> bytes | None:
        return self.memory.get(key, default)

    def set(self, key: str, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")
        self.memory[key] = value

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        return {key: val for key in args if (val := self.memory.get(key)) is not None}

    def set_many(self, **kwargs: bytes) -> None:
        for key, value in kwargs.items():
            if not isinstance(value, bytes):
                raise TypeError(f"Expected bytes for {key}, got {type(value).__name__}")
        self.memory.update(kwargs)

    def items(self) -> Iterable[tuple[str, bytes]]:
        return self.memory.items()

    def __contains__(self, key: str) -> bool:
        return key in self.memory
