"""
Composite cache for composing N KVStores in a tiered hierarchy.

Uses a list of stores ordered from fastest (L1) to most durable (Ln).
Reads check each tier in order, populating faster tiers on cache miss.
Writes propagate to all tiers (slower tiers can be wrapped in WriteBehind).

Example:
    # Two-tier: local cache + remote dict
    kv = Composite([
        Disk("/tmp/agex-cache"),
        ModalDict("my-dict"),
    ])

    # Three-tier: local + remote dict + durable volume
    kv = Composite([
        Disk("/tmp/agex-cache"),
        ModalDict("my-dict"),
        WriteBehind(ModalFile("/vol/state")),
    ])
"""

from typing import Iterable, Mapping

from agex.state.kv.base import KVStore


class Composite(KVStore):
    """
    N-tier cache composing any number of KVStores.

    On get():
        1. Check L1, L2, ..., Ln in order
        2. On hit at tier i, populate L1..L(i-1) and return
        3. On miss at all tiers, return None

    On set()/set_many():
        1. Write to Ln (most durable, authoritative)
        2. Write to Ln-1, ..., L1 (best effort)

    On cas():
        1. Delegate to Ln (authoritative tier for atomicity)
        2. Update L1..Ln-1 on success

    Args:
        stores: List of KVStores ordered fastest → most durable.
                Must have at least one store.

    Example:
        stores = [
            Disk("/tmp"),              # L1: fast local cache
            ModalDict("state"),        # L2: fast remote, 7-day TTL
            WriteBehind(ModalFile()),  # L3: slow remote, forever
        ]
        kv = Composite(stores)
    """

    def __init__(self, stores: list[KVStore]):
        if not stores:
            raise ValueError("Composite requires at least one store")
        self._stores = stores
        self._hits = 0
        self._misses = 0

    # ---- Read operations ----

    def get(self, key: str) -> bytes | None:
        """Get from stores, checking each tier in order."""
        for i, store in enumerate(self._stores):
            try:
                value = store.get(key)
                if value is not None:
                    # Hit at tier i - populate all faster tiers
                    if i > 0:
                        self._hits += 1
                        for j in range(i):
                            try:
                                self._stores[j].set(key, value)
                            except Exception:
                                pass  # Cache population failure is non-fatal
                    return value
            except Exception:
                # Store failure - try next tier
                continue

        # Miss at all tiers
        self._misses += 1
        return None

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        """Get multiple keys with multi-tier cache check."""
        result = {}
        remaining = set(args)

        for i, store in enumerate(self._stores):
            if not remaining:
                break

            try:
                # Try to fetch all remaining keys from this tier
                tier_values = {}
                for key in remaining:
                    value = store.get(key)
                    if value is not None:
                        tier_values[key] = value

                # For each hit, populate faster tiers
                if tier_values and i > 0:
                    self._hits += len(tier_values)
                    for j in range(i):
                        try:
                            self._stores[j].set_many(**tier_values)
                        except Exception:
                            pass

                # Add to result and remove from remaining
                result.update(tier_values)
                remaining -= tier_values.keys()

            except Exception:
                continue

        # Count final misses
        self._misses += len(remaining)
        return result

    def __contains__(self, key: str) -> bool:
        """Check if key exists in any tier."""
        for store in self._stores:
            try:
                if key in store:
                    return True
            except Exception:
                continue
        return False

    def keys(self) -> Iterable[str]:
        """Delegate to most durable tier (authoritative)."""
        return self._stores[-1].keys()

    def items(self) -> Iterable[tuple[str, bytes]]:
        """Delegate to most durable tier (authoritative)."""
        return self._stores[-1].items()

    # ---- Write operations ----

    def set(self, key: str, value: bytes) -> None:
        """Write to all tiers (most durable first)."""
        # Write to authoritative tier first (must succeed)
        self._stores[-1].set(key, value)

        # Best effort write to cache tiers
        for store in self._stores[:-1]:
            try:
                store.set(key, value)
            except Exception:
                pass  # Cache failure doesn't fail the write

    def set_many(self, **kwargs: bytes) -> None:
        """Write to all tiers (most durable first)."""
        # Write to authoritative tier first (must succeed)
        self._stores[-1].set_many(**kwargs)

        # Best effort write to cache tiers
        for store in self._stores[:-1]:
            try:
                store.set_many(**kwargs)
            except Exception:
                pass

    def remove(self, key: str) -> None:
        """Remove from all tiers (most durable first)."""
        self._stores[-1].remove(key)

        for store in self._stores[:-1]:
            try:
                store.remove(key)
            except Exception:
                pass

    def remove_many(self, *keys: str) -> None:
        """Remove multiple keys from all tiers."""
        self._stores[-1].remove_many(*keys)

        for store in self._stores[:-1]:
            try:
                store.remove_many(*keys)
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all tiers (most durable first)."""
        self._stores[-1].clear()

        for store in self._stores[:-1]:
            try:
                store.clear()
            except Exception:
                pass

    # ---- Atomic operations ----

    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """CAS on most durable tier, update caches on success."""
        success = self._stores[-1].cas(key, value, expected)
        if success:
            # Update cache tiers on successful CAS
            for store in self._stores[:-1]:
                try:
                    store.set(key, value)
                except Exception:
                    pass
        return success

    # ---- Stats ----

    def print_stats(self) -> None:
        """Print cache hit/miss stats."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        print("\n=== Composite Stats ===")
        print(f"  Tiers: {len(self._stores)}")
        print(f"  Hits: {self._hits}")
        print(f"  Misses: {self._misses}")
        print(f"  Hit rate: {hit_rate:.1f}%")
