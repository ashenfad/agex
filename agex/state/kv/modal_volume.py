"""
Modal Volume-backed KV store.

Uses Modal Volumes for persistent file-based storage. Each key is stored
as a file on the volume, providing durable storage without TTL expiration.

Designed for use as the durability tier in a Composite cache:
    Composite([
        Disk("/tmp"),           # L1: fast local
        ModalDict("state"),     # L2: fast remote, 7-day TTL
        WriteBehind(Volume()),  # L3: slow remote, forever
    ])
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from kvgit.kv import KVStore

if TYPE_CHECKING:
    import modal


# Debug flag - set to True to enable timing output
DEBUG = False


class Volume(KVStore):
    """
    KV store backed by a Modal Volume.

    Each key is stored as a file at `{mount_path}/{prefix}/{key}`.
    The volume must be mounted in the Modal container.

    Writes are committed immediately via set_many() which batches
    the volume.commit() call. For async writes, wrap in WriteBehind.

    Args:
        volume_name: Modal volume name (created if missing)
        mount_path: Local mount point for the volume (default: /vol)
        prefix: Optional prefix for key namespacing (creates subdirectory)

    Example:
        kv = Volume("agex-state", prefix="session-123")
        kv.set_many(counter=b"42", name=b"test")  # Batched commit
    """

    def __init__(
        self,
        volume_name: str,
        mount_path: str = "/vol",
        prefix: str = "",
    ):
        import modal

        self._volume: "modal.Volume" = modal.Volume.from_name(
            volume_name, create_if_missing=True
        )
        self._mount_path = Path(mount_path)
        self._prefix = prefix
        self._needs_reload = True

        # Ensure prefix directory exists
        self._base_path = self._mount_path / prefix if prefix else self._mount_path

        # Debug stats
        self._stats = {
            "get_count": 0,
            "get_time": 0.0,
            "set_count": 0,
            "set_time": 0.0,
            "reload_count": 0,
            "commit_count": 0,
        }

    def _log(self, op: str, duration: float, extra: str = "") -> None:
        if DEBUG:
            print(f"[Volume] {op}: {duration * 1000:.1f}ms {extra}")

    def _encode_key(self, key: str) -> str:
        """Encode key for safe filesystem use."""
        # Replace path separators to prevent directory traversal
        return key.replace("/", "__").replace("\\", "__")

    def _decode_key(self, filename: str) -> str:
        """Decode filesystem name back to key."""
        return filename.replace("__", "/")

    def _key_path(self, key: str) -> Path:
        """Get full path for a key."""
        return self._base_path / self._encode_key(key)

    def _ensure_reload(self) -> None:
        """Reload volume if needed."""
        if self._needs_reload:
            start = time.perf_counter()
            self._volume.reload()
            duration = time.perf_counter() - start
            self._stats["reload_count"] += 1
            self._log("reload", duration)
            self._needs_reload = False

    def _ensure_base_path(self) -> None:
        """Ensure prefix directory exists."""
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _commit(self) -> None:
        """Commit changes to volume."""
        start = time.perf_counter()
        self._volume.commit()
        duration = time.perf_counter() - start
        self._stats["commit_count"] += 1
        self._log("commit", duration)
        # After commit, other containers may have written, so reload needed
        self._needs_reload = True

    # ---- Read operations ----

    def _raw_get(self, key: str) -> bytes | None:
        """Read file without reload (internal use after _ensure_reload)."""
        start = time.perf_counter()
        path = self._key_path(key)

        try:
            if path.exists():
                value = path.read_bytes()
                duration = time.perf_counter() - start
                self._stats["get_count"] += 1
                self._stats["get_time"] += duration
                self._log("get", duration, f"key={key[:30]}...")
                return value
        except Exception:
            pass

        return None

    def get(self, key: str) -> bytes | None:
        """Get bytes value for key, or None if not found."""
        self._ensure_reload()
        return self._raw_get(key)

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        """Get multiple keys, returning only keys that exist."""
        self._ensure_reload()
        result = {}
        for key in args:
            value = self._raw_get(key)  # Use _raw_get to avoid redundant reloads
            if value is not None:
                result[key] = value
        return result

    def __contains__(self, key: str) -> bool:
        """Check if key exists in store."""
        self._ensure_reload()
        return self._key_path(key).exists()

    def keys(self) -> Iterable[str]:
        """Iterate over all keys."""
        self._ensure_reload()
        if not self._base_path.exists():
            return

        for path in self._base_path.iterdir():
            if path.is_file():
                yield self._decode_key(path.name)

    def items(self) -> Iterable[tuple[str, bytes]]:
        """Iterate over all key-value pairs."""
        self._ensure_reload()
        if not self._base_path.exists():
            return

        for path in self._base_path.iterdir():
            if path.is_file():
                try:
                    yield self._decode_key(path.name), path.read_bytes()
                except Exception:
                    continue

    # ---- Write operations ----

    def set(self, key: str, value: bytes) -> None:
        """Set bytes value for key. Commits immediately."""
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")

        self._ensure_reload()
        self._ensure_base_path()

        start = time.perf_counter()
        path = self._key_path(key)
        path.write_bytes(value)

        duration = time.perf_counter() - start
        self._stats["set_count"] += 1
        self._stats["set_time"] += duration
        self._log("set", duration, f"key={key[:30]}... size={len(value)}")

        self._commit()

    def set_many(self, **kwargs: bytes) -> None:
        """Set multiple key-value pairs with single commit."""
        if not kwargs:
            return

        self._ensure_reload()
        self._ensure_base_path()

        for key, value in kwargs.items():
            if not isinstance(value, bytes):
                raise TypeError(f"Expected bytes for {key}, got {type(value).__name__}")
            path = self._key_path(key)
            path.write_bytes(value)

        # Single commit for all writes
        self._commit()

    def remove(self, key: str) -> None:
        """Remove a key if present."""
        self._ensure_reload()
        path = self._key_path(key)
        try:
            path.unlink()
            self._commit()
        except FileNotFoundError:
            pass

    def remove_many(self, *keys: str) -> None:
        """Remove multiple keys."""
        self._ensure_reload()
        removed_any = False
        for key in keys:
            try:
                self._key_path(key).unlink()
                removed_any = True
            except FileNotFoundError:
                pass

        if removed_any:
            self._commit()

    def clear(self) -> None:
        """Remove all items from the store."""
        self._ensure_reload()
        if self._base_path.exists():
            removed_any = False
            for path in self._base_path.iterdir():
                if path.is_file():
                    path.unlink()
                    removed_any = True
            if removed_any:
                self._commit()

    # ---- Atomic operations ----

    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """
        Simple read-compare-write CAS.

        Note: Not atomic across distributed containers. For true atomicity,
        use ModalDict as the CAS tier in a Composite cache.
        """
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")

        self._ensure_reload()
        self._ensure_base_path()

        path = self._key_path(key)

        # Read current value
        try:
            current = path.read_bytes() if path.exists() else None
        except Exception:
            current = None

        # Compare
        if expected is None:
            # Create-if-missing: fail if exists
            if current is not None:
                return False
        else:
            # Update-if-matches: fail if mismatch
            if current != expected:
                return False

        # Write and commit
        path.write_bytes(value)
        self._commit()
        return True

    def print_stats(self) -> None:
        """Print accumulated stats."""
        print("\n=== Volume Stats ===")
        for k, v in self._stats.items():
            print(f"  {k}: {v}")
