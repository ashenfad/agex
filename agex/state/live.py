"""Live: immediate-write ephemeral state with no versioning."""

from collections.abc import Iterator, MutableMapping
from typing import Any


class Live(MutableMapping[str, Any]):
    """Immediate-write in-memory store.

    Used as a typed marker for ephemeral, non-versioned state.
    Distinguishable from ``Staged`` via ``isinstance`` checks.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_many(self, *keys: str) -> dict[str, Any]:
        return {k: self._data[k] for k in keys if k in self._data}

    def keys(self) -> set[str]:  # type: ignore[override]
        return set(self._data.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]
