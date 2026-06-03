"""Agent-session-scoped cache: a persistent dict for the agent.

Wraps any state-like ``MutableMapping`` with a fixed prefix so cache
keys live in their own slice of state.  Sub-agent isolation comes
from the surrounding ``Namespaced`` state wrapper: each sub-agent's
cache lives at ``<agent_namespace>/__cache__/<key>``.

The cache is injected into the agent's namespace by ``build_namespace``
on every emission.  Values are validated for picklability at write
time using stdlib ``pickle.HIGHEST_PROTOCOL`` — the protocol the
state codec uses underneath.  The codec is kvgit's ``scientific``
preset (``ChunkingPickler`` + ``NumpyCodec``); for non-array values
the chunking pickler falls through to the same protocol, so anything
``pickle.dumps`` rejects the codec also rejects (and would silently
marker), and anything ``pickle.dumps`` accepts the codec accepts
(and may externalize more efficiently).  The validator gives the
agent an immediate ``CacheError`` instead of a silent
``UnpicklableMarker`` discovered on a later read.

The cache holds **data**.  Sandbox-defined functions and classes are
plain Python objects (no importable module) and are not picklable, so
they cannot be cached — cache their *results* instead, and put reusable
*code* under ``helpers/`` (source, re-imported on demand).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator, MutableMapping
from typing import Any

PREFIX = "__cache__/"


class CacheError(ValueError):
    """Raised when a cache operation cannot complete (e.g. unpicklable value)."""


class Cache(MutableMapping[str, Any]):
    """A persistent dict for the agent, scoped to the agent's session.

    Keys must be plain strings.  Two patterns are reserved at write time:
      - keys beginning with ``__`` (framework bookkeeping convention)
      - keys containing ``/`` (used for state namespacing)

    Both raise :class:`ValueError`.  Values must be picklable data;
    picklability is checked at ``cache[k] = v`` time.
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
        # Match dict's wrong-shape vs missing distinction: ``dict``
        # raises TypeError for unhashable keys and KeyError for
        # missing-but-hashable keys.  Cache's stricter constraint
        # ("must be str") follows the same split — non-str raises
        # TypeError; str-but-missing raises KeyError via the state.
        if not isinstance(key, str):
            raise TypeError(f"Cache keys must be strings, got {type(key).__name__}")
        return self._state[PREFIX + key]

    def __setitem__(self, key: str, value: Any) -> None:
        qualified = self._check_writable_key(key)
        # Validate picklability up front so the agent gets a clear
        # error at write time rather than discovering a silent marker
        # on a later read.  ``pickle.HIGHEST_PROTOCOL`` matches what
        # the state codec uses underneath; anything stdlib pickle
        # rejects, the codec also rejects.  Sandbox-defined functions /
        # classes are plain (unpicklable) objects and fail here — cache
        # data, not code.
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise CacheError(
                f"Cannot cache key {key!r}: value is not picklable ({exc}). "
                f"The cache holds data; for reusable code use helpers/."
            ) from exc
        self._state[qualified] = value

    def __delitem__(self, key: str) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Cache keys must be strings, got {type(key).__name__}")
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


class RemoteCache(MutableMapping[str, Any]):
    """Worker-side cache facade backed by a parent-process ``Cache``
    via RPC.

    Used under process / kernel isolation: the bridge layer injects an
    :class:`sandtrap.RpcProxyMarker` (with ``wrapper="agex.cache:
    RemoteCache"``) into the namespace; the worker substitutes it
    with this class wrapping an ``RpcProxy``.  Method calls translate
    into RPC round-trips that reach the parent's live ``Cache(state)``,
    so writes propagate to the agent's real session cache and reads
    see whatever the parent has cached.  Cache values are picklable
    data, so they cross the RPC boundary unchanged.
    """

    def __init__(self, proxy: Any) -> None:
        self._proxy = proxy

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Cache keys must be strings, got {type(key).__name__}")
        return self._proxy._call("getitem", key)

    def __setitem__(self, key: str, value: Any) -> None:
        # Validation (key shape, picklability) happens on the parent
        # via Cache.__setitem__.  Errors propagate back as RPC
        # exceptions and re-raise here.
        self._proxy._call("setitem", key, value)

    def __delitem__(self, key: str) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Cache keys must be strings, got {type(key).__name__}")
        self._proxy._call("delitem", key)

    def __iter__(self) -> Iterator[str]:
        # Parent returns a list snapshot; iterate locally.
        return iter(self._proxy._call("iter"))

    def __len__(self) -> int:
        return self._proxy._call("len")

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._proxy._call("contains", key)

    def __repr__(self) -> str:
        try:
            keys = sorted(self._proxy._call("iter"))
        except Exception:
            return "Cache(<unreadable>)"
        return f"Cache({keys!r})"
