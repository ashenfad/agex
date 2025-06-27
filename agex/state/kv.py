from abc import ABC, abstractmethod
from typing import Iterable, Mapping, cast

from diskcache import Cache as DiskCache


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
    def __contains__(self, key: str) -> bool:
        """Check if key exists in store."""
        pass


class Memory(KVStore):
    """A memory-backed KV store that stores values as bytes."""

    def __init__(self):
        self.memory: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.memory.get(key)

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


SIXTY_FOUR_MB = 64 * 1024 * 1024
ONE_GB = 1024 * 1024 * 1024


class Cache(KVStore):
    """A write-through cache that stores values in memory."""

    def __init__(self, store: KVStore, max_bytes: int = SIXTY_FOUR_MB):
        self.cache: dict[str, bytes] = {}
        self.store = store
        self.max_bytes = max_bytes

    def _evict(self) -> None:
        total = sum(len(v) for v in self.cache.values())
        while total > self.max_bytes and self.cache:
            key, value = next(iter(self.cache.items()))
            total -= len(value)
            del self.cache[key]

    def get(self, key: str) -> bytes | None:
        if key in self.cache:
            return self.cache[key]

        miss = self.store.get(key)
        if miss is not None:
            self.cache[key] = miss
            self._evict()
        return miss

    def set(self, key: str, value: bytes) -> None:
        self.cache[key] = value
        self.store.set(key, value)
        self._evict()

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        hits = {k: self.cache[k] for k in args if k in self.cache}
        misses = self.store.get_many(*(set(args) - set(hits)))
        self.cache.update(misses)
        self._evict()
        return hits | dict(misses)

    def set_many(self, **kwargs: bytes) -> None:
        self.cache.update(kwargs)
        self.store.set_many(**kwargs)
        self._evict()

    def items(self) -> Iterable[tuple[str, bytes]]:
        return self.store.items()

    def __contains__(self, key: str) -> bool:
        return key in self.cache or key in self.store


class Disk(KVStore):
    def __init__(self, directory: str, size_limit: int = ONE_GB):
        self.store = DiskCache(directory, size_limit=size_limit)

    def clear(self) -> None:
        self.store.clear()

    def get(self, key: str) -> bytes | None:
        return cast(bytes | None, self.store.get(key))

    def set(self, key: str, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")
        self.store[key] = value

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        # Could be optimized with batch operations if DiskCache supports it
        return {k: v for k in args if (v := self.get(k)) is not None}

    def set_many(self, **kwargs) -> None:  # Removed problematic type hint
        # Validate all values first to ensure atomicity
        for key, value in kwargs.items():
            if not isinstance(value, bytes):
                raise TypeError(f"Expected bytes for {key}, got {type(value).__name__}")

        # Only set if all values are valid
        for key, value in kwargs.items():
            self.set(key, value)

    def items(self) -> Iterable[tuple[str, bytes]]:
        for key in self.store.iterkeys():
            yield str(key), cast(bytes, self.store[key])

    def __contains__(self, key: str) -> bool:
        return key in self.store
