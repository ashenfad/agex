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

Sandbox-defined ``StFunction`` / ``StClass`` wrappers define
``__getstate__`` / ``__setstate__`` and pass validation; lambdas and
other locally-defined functions outside the sandbox do not.  Cached
wrappers are re-activated on every ``exec`` via sandtrap's
``__sandtrap_activate__`` container hook, so a cached helper from
one task remains callable in any later task.

To keep the activation hook from decoding every cache value on every
emission (data-only caches are common; full decode is wasteful),
the Cache maintains a side index of which keys hold sandbox-defined
wrappers.  The hook walks only those keys; data values are decoded
lazily on access.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator, MutableMapping
from typing import Any

PREFIX = "__cache__/"

# Side-index key tracking which cache keys hold sandbox-defined
# wrappers (StFunction / StClass / StInstance).  Lives outside the
# ``__cache__/`` prefix so it never appears in user-facing cache
# iteration.  See ``__sandtrap_activate__`` for the read path.
_WRAPPER_INDEX_KEY = "__cache_wrappers__"


def _is_sandtrap_wrapper(value: Any) -> bool:
    """Return True if ``value`` is a sandtrap wrapper that may need
    re-activation when retrieved from the cache."""
    # Imported lazily so the module loads cleanly even if the caller
    # has stubbed sandtrap (e.g. tests using only Live state).
    from sandtrap.wrappers import StClass, StFunction, StInstance

    return isinstance(value, (StFunction, StClass, StInstance))


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
        # rejects, the codec also rejects.
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise CacheError(
                f"Cannot cache key {key!r}: value is not picklable ({exc})"
            ) from exc
        # Maintain the wrapper-index so the activation hook can find
        # sandbox-defined values without decoding the entire cache.
        wrappers = self._wrapper_keys()
        if _is_sandtrap_wrapper(value):
            if key not in wrappers:
                wrappers.add(key)
                self._save_wrapper_keys(wrappers)
        elif key in wrappers:
            # The slot used to hold a wrapper; the new value is plain
            # data, so drop it from the index.
            wrappers.discard(key)
            self._save_wrapper_keys(wrappers)
        self._state[qualified] = value

    def __delitem__(self, key: str) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Cache keys must be strings, got {type(key).__name__}")
        wrappers = self._wrapper_keys()
        if key in wrappers:
            wrappers.discard(key)
            self._save_wrapper_keys(wrappers)
        del self._state[PREFIX + key]

    def _wrapper_keys(self) -> set[str]:
        """Read the wrapper-index from state.  Returns an empty set
        when the index hasn't been written yet (data-only cache)."""
        return set(self._state.get(_WRAPPER_INDEX_KEY) or ())

    def _save_wrapper_keys(self, keys: set[str]) -> None:
        """Persist the wrapper-index, or drop it when empty so the
        state stays clean for caches that have never held a wrapper."""
        if keys:
            self._state[_WRAPPER_INDEX_KEY] = keys
        elif _WRAPPER_INDEX_KEY in self._state:
            del self._state[_WRAPPER_INDEX_KEY]

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

    def __sandtrap_activate__(self, activate_value, gates, sandbox, namespace) -> None:
        """Sandtrap container-activation hook.

        Sandtrap calls this on every ``exec`` after building the
        namespace.  We walk only the cache keys that hold
        sandbox-defined wrappers (tracked in the wrapper-index, kept
        in sync at write time) and re-activate them so that a
        ``StFunction`` cached in one task remains callable when
        retrieved in a later task.  Plain data values are not
        decoded — they're loaded lazily when the agent reads them.

        ``namespace`` is the top-level exec namespace, passed through
        to ``activate_value`` so a cached wrapper that references a
        name resolved in the top-level namespace (e.g. a registered
        function or another cached helper) can find it during
        late-binding rebuild.

        Errors per-value are swallowed: a single stale wrapper or a
        deserialization issue must not break ``exec``.
        """
        for key in self._wrapper_keys():
            try:
                val = self[key]
            except Exception:
                continue
            try:
                activate_value(val, gates, sandbox=sandbox, namespace=namespace)
            except Exception:
                continue


class RemoteCache(MutableMapping[str, Any]):
    """Worker-side cache facade backed by a parent-process ``Cache``
    via RPC.

    Used under process / kernel isolation: the bridge layer injects an
    :class:`sandtrap.RpcProxyMarker` (with ``wrapper="agex.cache:
    RemoteCache"``) into the namespace; the worker substitutes it
    with this class wrapping an ``RpcProxy``.  Method calls translate
    into RPC round-trips that reach the parent's live ``Cache(state)``,
    so writes propagate to the agent's real session cache and reads
    see whatever the parent has cached.

    Sandbox-defined wrappers (``StFunction`` / ``StClass``) are
    re-activated locally with the worker's gates on every read — they
    arrive inactive because pickle strips the sandbox-bound
    ``_compiled`` / ``_sandbox`` / ``_gates`` references on the
    parent-side serialization.  The activation hook captures the
    worker's gates on first ``exec`` so subsequent ``__getitem__``
    calls have what they need.
    """

    def __init__(self, proxy: Any) -> None:
        self._proxy = proxy
        self._gates: Any = None
        self._sandbox: Any = None

    def __sandtrap_activate__(self, activate_value, gates, sandbox, namespace) -> None:
        # Stash the worker's gates / sandbox so __getitem__ can
        # activate inactive wrappers it pulls back from the parent.
        # We don't pre-warm cache reads here — there's no benefit
        # and it'd waste IPC on values the agent never touches.
        self._gates = gates
        self._sandbox = sandbox

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Cache keys must be strings, got {type(key).__name__}")
        val = self._proxy._call("getitem", key)
        # Activate any inactive wrapper the parent shipped over.
        # ``activate_value`` short-circuits cheaply for non-wrappers,
        # so plain data values pay only an isinstance check.
        if self._gates is not None:
            from sandtrap.wrappers import activate_value

            try:
                activate_value(val, self._gates, sandbox=self._sandbox)
            except Exception:
                pass
        return val

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
