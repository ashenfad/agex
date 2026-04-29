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


def _make_local_cache() -> "Cache":
    """Constructor used by ``Cache.__reduce__`` to materialize a fresh
    in-memory Cache on the receiving end of a pickle round-trip
    (typically a process-isolated subprocess)."""
    return Cache({})


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
            raise KeyError(key)
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

    def __reduce__(self):
        """Pickle support for process-isolated emissions.

        Versioned ``state`` holds non-picklable resources (kvgit
        threading locks etc.), so a Cache wrapping it can't cross a
        process boundary as-is.  Without this hook sandtrap's
        ``filter_namespace`` would drop the cache from the
        subprocess's namespace with a warning — the agent would then
        see a ``NameError`` instead of a usable ``cache``.

        We probe the underlying state's picklability:
          - If it pickles (e.g. a plain dict, ``Live`` state), we
            preserve the round-trip — the receiving end gets a
            ``Cache`` over an equivalent state, with the same data.
          - If not (e.g. Staged with kvgit locks), we fall back to a
            fresh empty Cache so the subprocess gets a working
            object.

        Limitation in the fallback case: writes performed inside a
        process-isolated emission are stored in the subprocess's
        local copy and do NOT propagate back to the parent's session
        cache.  Reads of keys the parent cached are also unavailable.
        Cross-process cache durability would require explicit IPC,
        which the bridge doesn't currently do.  Most workloads use
        ``isolation='none'`` (the default), where cache works as
        documented.
        """
        try:
            pickle.dumps(self._state, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return (_make_local_cache, ())
        return (Cache, (self._state,))

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
