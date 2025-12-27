"""
Tiered cache for composing two KVStores.

Uses a fast local cache (any KVStore) as L1, falling back to a remote
source (any KVStore) on cache miss. Writes go to both.

Example:
    kv = TieredCache(
        cache=Disk("/tmp/agex-cache"),
        source=ModalDict("my-dict"),
    )
"""

from typing import Iterable, Mapping

from agex.state.kv.base import KVStore


class TieredCache(KVStore):
    """
    Two-tier cache composing any two KVStores.

    On get():
        1. Check cache (L1)
        2. If miss, fetch from source (L2) and cache locally

    On set()/set_many():
        1. Write to source (L2, authoritative)
        2. Write to cache (L1)

    On cas():
        1. Delegate to source (L2, atomic operation)
        2. Update cache (L1) on success

    Args:
        cache: Fast local KVStore (e.g., Disk, Memory)
        source: Authoritative remote KVStore (e.g., ModalDict)
    """

    def __init__(self, cache: KVStore, source: KVStore):
        self._cache = cache
        self._source = source
        self._hits = 0
        self._misses = 0

    # ---- Read operations ----

    def get(self, key: str) -> bytes | None:
        """Get from cache, falling back to source."""
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached

        # Cache miss - fetch from source
        self._misses += 1
        value = self._source.get(key)
        if value is not None:
            try:
                self._cache.set(key, value)
            except Exception:
                pass  # Cache failure doesn't fail the read
        return value

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        """Get multiple keys with cache check."""
        result = {}
        source_needed = []

        for key in args:
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                result[key] = cached
            else:
                source_needed.append(key)

        if source_needed:
            self._misses += len(source_needed)
            source_values = self._source.get_many(*source_needed)
            for key, value in source_values.items():
                try:
                    self._cache.set(key, value)
                except Exception:
                    pass  # Cache failure doesn't fail the read
                result[key] = value

        return result

    def __contains__(self, key: str) -> bool:
        """Check cache first, then source."""
        if key in self._cache:
            return True
        return key in self._source

    def keys(self) -> Iterable[str]:
        """Delegate to source (cache may have stale/partial data)."""
        return self._source.keys()

    def items(self) -> Iterable[tuple[str, bytes]]:
        """Delegate to source."""
        return self._source.items()

    # ---- Write operations ----

    def set(self, key: str, value: bytes) -> None:
        """Write to source and cache."""
        self._source.set(key, value)  # Critical - must succeed
        try:
            self._cache.set(key, value)  # Best effort
        except Exception:
            pass  # Cache failure doesn't fail the write

    def set_many(self, **kwargs: bytes) -> None:
        """Write to source and cache."""
        self._source.set_many(**kwargs)  # Critical - must succeed
        try:
            self._cache.set_many(**kwargs)  # Best effort
        except Exception:
            pass  # Cache failure doesn't fail the write

    def remove(self, key: str) -> None:
        """Remove from source and cache."""
        self._source.remove(key)  # Critical - must succeed
        try:
            self._cache.remove(key)  # Best effort
        except Exception:
            pass  # Cache failure doesn't fail the remove

    def remove_many(self, *keys: str) -> None:
        """Remove multiple keys."""
        self._source.remove_many(*keys)  # Critical - must succeed
        try:
            self._cache.remove_many(*keys)  # Best effort
        except Exception:
            pass  # Cache failure doesn't fail the remove

    def clear(self) -> None:
        """Clear source and cache."""
        self._source.clear()  # Critical - must succeed
        try:
            self._cache.clear()  # Best effort
        except Exception:
            pass  # Cache failure doesn't fail the clear

    # ---- Atomic operations ----

    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """CAS on source, update cache on success."""
        success = self._source.cas(key, value, expected)
        if success:
            try:
                self._cache.set(key, value)
            except Exception:
                pass  # Cache failure doesn't fail the CAS
        return success

    # ---- Stats ----

    def print_stats(self) -> None:
        """Print cache hit/miss stats."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        print("\n=== TieredCache Stats ===")
        print(f"  Hits: {self._hits}")
        print(f"  Misses: {self._misses}")
        print(f"  Hit rate: {hit_rate:.1f}%")
