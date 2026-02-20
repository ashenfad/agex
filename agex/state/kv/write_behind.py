import queue
import threading
from typing import Iterable, Mapping

from kvit.kv import KVStore


class WriteBehind(KVStore):
    """
    A write-behind wrapper that pushes writes to a background thread.

    This is useful for masking the latency of slow storage backends (like S3 or
    remote databases) by returning control to the agent immediately.
    """

    def __init__(self, store: KVStore):
        self.store = store
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break

            func_name, args, kwargs = item
            try:
                getattr(self.store, func_name)(*args, **kwargs)
            except Exception as e:
                # We can't raise to the caller, so we log to stderr
                import sys

                print(f"WriteBehind error ({func_name}): {e}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def get(self, key: str) -> bytes | None:
        self.flush()
        return self.store.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._queue.put(("set", (key, value), {}))

    def get_many(self, *args: str) -> Mapping[str, bytes]:
        self.flush()
        return self.store.get_many(*args)

    def set_many(self, **kwargs: bytes) -> None:
        self._queue.put(("set_many", (), kwargs))

    def items(self) -> Iterable[tuple[str, bytes]]:
        self.flush()
        return self.store.items()

    def keys(self) -> Iterable[str]:
        self.flush()
        return self.store.keys()

    def __contains__(self, key: str) -> bool:
        self.flush()
        return key in self.store

    def remove(self, key: str) -> None:
        self._queue.put(("remove", (key,), {}))

    def remove_many(self, *keys: str) -> None:
        self._queue.put(("remove_many", keys, {}))

    def flush(self) -> None:
        """Wait for all pending writes to complete."""
        self._queue.join()

    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """
        CAS requires synchronous execution - flush pending writes first.

        This ensures we're comparing against the true current value,
        not a value that has pending writes in the queue.
        """
        self.flush()
        return self.store.cas(key, value, expected)

    def clear(self) -> None:
        """Flush pending writes and clear the underlying store."""
        self.flush()
        self.store.clear()
