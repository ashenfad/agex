from typing import Iterable, Mapping, cast

from diskcache import Cache as DiskCache

from agex.state.kv.base import KVStore

ONE_GB = 1024 * 1024 * 1024


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
        with self.store.transact():
            for key, value in kwargs.items():
                self.set(key, value)

    def items(self) -> Iterable[tuple[str, bytes]]:
        for key in self.store.iterkeys():
            yield str(key), cast(bytes, self.store[key])

    def keys(self) -> Iterable[str]:
        for key in self.store.iterkeys():
            yield str(key)

    def __contains__(self, key: str) -> bool:
        return key in self.store

    def remove(self, key: str) -> None:
        try:
            del self.store[key]
        except KeyError:
            pass

    def remove_many(self, *keys: str) -> None:
        with self.store.transact():
            for key in keys:
                self.store.delete(key, retry=False)

    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """Atomic compare-and-swap using diskcache transactions."""
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")

        with self.store.transact():
            current = cast(bytes | None, self.store.get(key))
            if current == expected:
                self.store[key] = value
                return True
            return False
