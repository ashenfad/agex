"""
Modal Dict-backed KV store.

Uses Modal's native Dict service for distributed key-value storage,
providing lower latency than volume-based ModalFile and atomic
locking via `skip_if_exists`.

Note: Modal Dict entries expire after 7 days of inactivity. Reads
refresh the TTL, so active sessions stay alive indefinitely. Inactive
sessions auto-expire (useful for cleanup, but be aware for long-running
state like event logs).
"""

import time
from typing import TYPE_CHECKING, Iterable, Mapping

from kvit.kv import KVStore

if TYPE_CHECKING:
    import modal


# Debug flag - set to True to enable timing output
DEBUG = False


class ModalDict(KVStore):
    """
    KV store backed by Modal's native Dict service.

    Modal Dict provides distributed key-value storage with:
    - ~10-50ms latency per operation (single RPC, no reload/commit dance)
    - Atomic create-if-missing via `skip_if_exists`
    - Automatic 7-day TTL on inactivity (reads refresh TTL)
    - Unlimited items
    - cloudpickle serialization (same as agex)

    Args:
        name: Modal Dict name. Created if missing.
        prefix: Optional prefix for key namespacing (e.g., session ID).
                Keys are stored as "{prefix}:{key}" if prefix is set.

    Example:
        kv = ModalDict(name="agex-state", prefix="session-123")
        kv.set("counter", b"42")
        value = kv.get("counter")  # b"42"
    """

    def __init__(self, name: str, prefix: str = ""):
        import modal

        self._dict: "modal.Dict" = modal.Dict.from_name(name, create_if_missing=True)
        self._prefix = prefix

        # Debug stats
        self._stats = {
            "get_count": 0,
            "get_time": 0.0,
            "set_count": 0,
            "set_time": 0.0,
            "set_many_count": 0,
            "set_many_time": 0.0,
            "cas_count": 0,
            "cas_time": 0.0,
            "contains_count": 0,
            "contains_time": 0.0,
        }

    def _log(self, op: str, duration: float, extra: str = ""):
        if DEBUG:
            count = self._stats.get(f"{op.split('(')[0]}_count", "?")
            print(f"[ModalDict] #{count} {op}: {duration*1000:.1f}ms {extra}")

    def print_stats(self):
        """Print accumulated stats."""
        print("\n=== ModalDict Stats ===")
        for k, v in self._stats.items():
            print(f"  {k}: {v}")
        print()

    def _key(self, key: str) -> str:
        """Build prefixed key for storage."""
        return f"{self._prefix}:{key}" if self._prefix else key

    def _unprefix(self, key: str) -> str:
        """Remove prefix from storage key."""
        if self._prefix and key.startswith(f"{self._prefix}:"):
            return key[len(self._prefix) + 1 :]
        return key

    # ---- Read operations ----

    def get(self, key: str) -> bytes | None:
        """Get bytes value for key, or None if not found."""
        start = time.perf_counter()
        result = self._dict.get(self._key(key))
        duration = time.perf_counter() - start
        self._stats["get_count"] += 1
        self._stats["get_time"] += duration
        # Log all gets to see the count grow
        self._log("get", duration, f"key={key[:40]}...")
        return result

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        """Get multiple keys, returning only keys that exist."""
        if not args:
            return {}
        # Modal Dict doesn't have batch get, so we iterate
        # Could be parallelized with asyncio in future
        result = {}
        for key in args:
            val = self.get(key)
            if val is not None:
                result[key] = val
        return result

    def __contains__(self, key: str) -> bool:
        """Check if key exists in store."""
        start = time.perf_counter()
        result = self._dict.contains(self._key(key))
        duration = time.perf_counter() - start
        self._stats["contains_count"] += 1
        self._stats["contains_time"] += duration
        if DEBUG and duration > 0.1:
            self._log("contains", duration, f"key={key[:30]}...")
        return result

    def keys(self) -> Iterable[str]:
        """Iterate over all keys (with prefix stripped)."""
        for k in self._dict.keys():
            # Only yield keys that match our prefix
            if isinstance(k, str):
                if self._prefix:
                    if k.startswith(f"{self._prefix}:"):
                        yield self._unprefix(k)
                else:
                    yield k

    def items(self) -> Iterable[tuple[str, bytes]]:
        """Iterate over all key-value pairs."""
        for k, v in self._dict.items():
            if isinstance(k, str):
                if self._prefix:
                    if k.startswith(f"{self._prefix}:"):
                        yield self._unprefix(k), v
                else:
                    yield k, v

    # ---- Write operations ----

    def set(self, key: str, value: bytes) -> None:
        """Set bytes value for key."""
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")
        start = time.perf_counter()
        self._dict[self._key(key)] = value
        duration = time.perf_counter() - start
        self._stats["set_count"] += 1
        self._stats["set_time"] += duration
        if DEBUG and duration > 0.1:
            self._log("set", duration, f"key={key[:30]}... size={len(value)}")

    def set_many(self, **kwargs: bytes) -> None:
        """Set multiple key-value pairs."""
        if not kwargs:
            return
        # Modal Dict has update() for batch writes
        prefixed = {self._key(k): v for k, v in kwargs.items()}
        start = time.perf_counter()
        self._dict.update(prefixed)
        duration = time.perf_counter() - start
        self._stats["set_many_count"] += 1
        self._stats["set_many_time"] += duration
        total_size = sum(len(v) for v in kwargs.values())
        self._log("set_many", duration, f"keys={len(kwargs)} total_size={total_size}")

    def remove(self, key: str) -> None:
        """Remove a key if present."""
        try:
            self._dict.pop(self._key(key))
        except KeyError:
            pass  # Already gone

    def remove_many(self, *keys: str) -> None:
        """Remove multiple keys."""
        for key in keys:
            self.remove(key)

    def clear(self) -> None:
        """Remove all items from the Dict."""
        # Note: This clears ALL items in the Dict, not just prefixed ones
        # For safety, only clear if no prefix (whole dict is ours)
        if not self._prefix:
            self._dict.clear()
        else:
            # Clear only our prefixed keys
            for key in list(self.keys()):
                self.remove(key)

    # ---- Atomic operations ----

    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """
        Atomic compare-and-swap operation.

        For expected=None (create-if-missing):
            Uses Modal's atomic `skip_if_exists` for true atomicity.

        For expected=bytes (update-if-matches):
            Uses read-compare-put pattern. NOT fully atomic, but matches
            the previous ModalFile behavior. The race window is small
            (~network RTT) and conflicts are rare for HEAD pointer updates.

        Args:
            key: The key to update
            value: The new value to set
            expected: The expected current value. None means "must not exist".

        Returns:
            True if swap succeeded, False otherwise.
        """
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")

        prefixed_key = self._key(key)
        start = time.perf_counter()

        if expected is None:
            # Create-if-missing: use atomic skip_if_exists
            result = self._dict.put(prefixed_key, value, skip_if_exists=True)
            duration = time.perf_counter() - start
            self._stats["cas_count"] += 1
            self._stats["cas_time"] += duration
            self._log("cas(create)", duration, f"key={key[:30]}... success={result}")
            return result

        # Update-if-matches: read-compare-put (not atomic)
        current = self._dict.get(prefixed_key)
        if current != expected:
            duration = time.perf_counter() - start
            self._stats["cas_count"] += 1
            self._stats["cas_time"] += duration
            self._log("cas(update)", duration, f"key={key[:30]}... MISMATCH")
            return False

        self._dict[prefixed_key] = value
        duration = time.perf_counter() - start
        self._stats["cas_count"] += 1
        self._stats["cas_time"] += duration
        self._log("cas(update)", duration, f"key={key[:30]}... success=True")
        return True
