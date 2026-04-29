"""Agent-session-scoped cache: a persistent dict for the agent.

Wraps any state-like ``MutableMapping`` with a fixed prefix so cache
keys live in their own slice of state.  Sub-agent isolation comes
from the surrounding ``Namespaced`` state wrapper: each sub-agent's
cache lives at ``<agent_namespace>/__cache__/<key>``.

The cache is injected into the agent's namespace by ``build_namespace``
on every emission.  Values are validated for picklability at write
time using cloudpickle — the same serializer the state codec uses —
so the agent gets an immediate, clear error rather than a silent
``UnpicklableMarker`` discovered on a later read.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any

import cloudpickle

PREFIX = "__cache__/"


class CacheError(ValueError):
    """Raised when a cache operation cannot complete (e.g. unpicklable value)."""


class Cache(MutableMapping[str, Any]):
    """A persistent dict for the agent, scoped to the agent's session.

    Keys must be plain strings.  Two patterns are reserved at write time:
      - keys beginning with ``__`` (framework bookkeeping convention)
      - keys containing ``/`` (used for state namespacing)

    Both raise :class:`ValueError`.  Values must be picklable; sandbox
    -defined functions and classes are fine.  Picklability errors raise
    from the underlying state's codec at ``cache[k] = v`` time.
    """

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self._state = state

    @staticmethod
    def _check_writable_key(key: Any) -> str:
        if not isinstance(key, str):
            raise TypeError(f"Cache keys must be strings, got {type(key).__name__}")
        if key.startswith("__"):
            raise ValueError(
                f"Cache keys may not start with '__' (reserved for framework): {key!r}"
            )
        if "/" in key:
            raise ValueError(
                f"Cache keys may not contain '/' (reserved for namespacing): {key!r}"
            )
        return PREFIX + key

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise KeyError(key)
        return self._state[PREFIX + key]

    def __setitem__(self, key: str, value: Any) -> None:
        qualified = self._check_writable_key(key)
        # Validate picklability up front so the agent gets a clear error
        # at write time rather than discovering a silent marker later.
        # cloudpickle is used because it matches the state codec's
        # capabilities (handles lambdas, sandbox-defined functions,
        # most Python objects).
        try:
            cloudpickle.dumps(value)
        except Exception as exc:
            raise CacheError(
                f"Cannot cache key {key!r}: value is not picklable ({exc})"
            ) from exc
        self._state[qualified] = value

    def __delitem__(self, key: str) -> None:
        if not isinstance(key, str):
            raise KeyError(key)
        del self._state[PREFIX + key]

    def __iter__(self) -> Iterator[str]:
        plen = len(PREFIX)
        for k in self._state.keys():
            if k.startswith(PREFIX):
                yield k[plen:]

    def __len__(self) -> int:
        return sum(1 for k in self._state.keys() if k.startswith(PREFIX))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return PREFIX + key in self._state

    def __repr__(self) -> str:
        try:
            keys = sorted(self)
        except Exception:
            return "Cache(<unreadable>)"
        return f"Cache({keys!r})"
