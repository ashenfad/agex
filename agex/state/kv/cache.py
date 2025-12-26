from typing import Iterable, Mapping

from agex.state.kv.base import KVStore

SIXTY_FOUR_MB = 64 * 1024 * 1024


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

    def keys(self) -> Iterable[str]:
        return self.store.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.cache or key in self.store

    def remove(self, key: str) -> None:
        self.cache.pop(key, None)
        self.store.remove(key)

    def remove_many(self, *keys: str) -> None:
        for key in keys:
            self.cache.pop(key, None)
        self.store.remove_many(*keys)

    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """Delegate CAS to underlying store and invalidate cache on success."""
        success = self.store.cas(key, value, expected)
        if success:
            # Update cache with new value
            self.cache[key] = value
            self._evict()
        else:
            # CAS failed - invalidate cache to force re-read
            self.cache.pop(key, None)
        return success
