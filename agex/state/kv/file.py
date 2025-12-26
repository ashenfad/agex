import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping

from agex.state.kv.base import KVStore


class File(KVStore):
    """
    A file-based KV store that uses one file per key.

    Designed for network filesystems (NFS, Modal Volumes) where SQLite locking
    is unreliable or unavailable.

    Args:
        directory: Root directory for the store.
        atomic_writes: If True, use temp file + rename for atomic safety (2x slower).
                      If False, use direct writes for performance (faster, but
                      readers may see partial data during writes).
        parallelism: Number of threads to use for bulk operations (set_many, get_many,
                     remove_many). Defaults to 10. Set to 1 to disable parallelism.
    """

    def __init__(
        self, directory: str, atomic_writes: bool = True, parallelism: int = 10
    ):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.atomic_writes = atomic_writes
        self.parallelism = parallelism
        self._executor = (
            ThreadPoolExecutor(max_workers=parallelism) if parallelism > 1 else None
        )

    def _key_path(self, key: str) -> Path:
        # Use safe encoding to prevent path traversal / invalid chars
        # We use urlsafe base64 to allow for specialized chars in keys
        safe_name = base64.urlsafe_b64encode(key.encode()).decode()
        return self.root / safe_name

    def get(self, key: str) -> bytes | None:
        path = self._key_path(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def set(self, key: str, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")

        path = self._key_path(key)

        if self.atomic_writes:
            # Atomic write: temp file + rename (2 network round trips)
            # Use pid/thread identifiers to avoid collisions in temp names
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                tmp.write_bytes(value)
                tmp.rename(path)  # Atomic on all POSIX filesystems
            except Exception:
                # Clean up temp file on error
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                raise
        else:
            # Direct write (1 network round trip, faster but may be partially visible)
            path.write_bytes(value)

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        if not args:
            return {}

        if self._executor:
            # Parallel fetch
            results = list(self._executor.map(self.get, args))
            return {k: v for k, v in zip(args, results) if v is not None}

        # Serial fetch
        result = {}
        for key in args:
            val = self.get(key)
            if val is not None:
                result[key] = val
        return result

    def set_many(self, **kwargs: bytes) -> None:
        if not kwargs:
            return

        if self._executor:
            # Parallel write
            # We must eagerly execute the iterator to consume exceptions
            list(self._executor.map(lambda item: self.set(*item), kwargs.items()))
        else:
            # Serial write
            for key, value in kwargs.items():
                self.set(key, value)

    def items(self) -> Iterable[tuple[str, bytes]]:
        for path in self.root.iterdir():
            if path.name.endswith(".tmp"):
                continue
            try:
                # Reverse base64 encoding to get original key
                # Note: this might fail if there are stray files in the directory
                key = base64.urlsafe_b64decode(path.name).decode()
                yield key, path.read_bytes()
            except Exception:
                continue

    def keys(self) -> Iterable[str]:
        for path in self.root.iterdir():
            if path.name.endswith(".tmp"):
                continue
            try:
                yield base64.urlsafe_b64decode(path.name).decode()
            except Exception:
                continue

    def __contains__(self, key: str) -> bool:
        return self._key_path(key).exists()

    def remove(self, key: str) -> None:
        try:
            self._key_path(key).unlink()
        except FileNotFoundError:
            pass

    def remove_many(self, *keys: str) -> None:
        if not keys:
            return

        if self._executor:
            list(self._executor.map(self.remove, keys))
        else:
            for key in keys:
                self.remove(key)

    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """
        Atomic Compare-And-Swap.

        Always uses temp file + rename to ensure that concurrent readers never
        see a partially written (corrupted) file. This ensures HEAD is always
        a valid commit hash, regardless of the `atomic_writes` setting.
        """
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes, got {type(value).__name__}")

        path = self._key_path(key)

        # 1. READ (Check)
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            current = None

        # 2. COMPARE
        if current != expected:
            return False

        # 3. WRITE (Always Atomic)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_bytes(value)
            tmp.rename(path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        return True
