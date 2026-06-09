"""
Bring-your-own state resolution.

A ``StateResolver`` owns the session → state lookup that the Local host
normally performs. The built-in storage modes give each session its own
substrate (a separate diskcache directory, a separate IndexedDB name),
which can't express "one shared substrate, a branch per session" — many
working trees over one repo, for concurrent sessions that still fork
cheaply. A custom resolver (e.g. one returning a ``Staged`` over a
shared ``VersionedKV`` pinned to a per-session branch) does.

Mirrors agex-ts's ``StateResolver`` (``agex-ts/src/state/connect.ts``),
introduced for agex-studio's concurrent-sessions work.

The resolver owns caching and lifecycle: the host delegates and does
not wrap resolver-produced states in its own session cache. Returning
the same instance for repeated ``resolve(session)`` calls is what keeps
the task loop, ``agent.fs()``, and ``agent.state()`` looking at one
staging area — return a fresh instance per call only if you want
independent working trees (e.g. optimistic concurrency within one
process).
"""

import re
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kvgit import Staged
    from kvgit.kv import KVStore

# Session ids are embedded into filesystem paths (disk storage) and
# IndexedDB names, so untrusted strings can escape the configured
# directory or namespace. Intentionally narrow — typical ids are
# "default" or "chat-<uuid>" style. Mirrors agex-ts's SAFE_SESSION_RE.
SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")


def assert_safe_session(session: str) -> None:
    """Reject session ids that could traverse paths or namespaces.

    Raises:
        ValueError: If the session id is empty, starts with ``.``, or
            contains characters outside ``[A-Za-z0-9_.-]``.
    """
    if not isinstance(session, str) or not SAFE_SESSION_RE.match(session):
        raise ValueError(
            f"invalid session id {session!r} — must match "
            f"{SAFE_SESSION_RE.pattern} to prevent path traversal in "
            f"storage backends"
        )


@runtime_checkable
class StateResolver(Protocol):
    """Per-session state lookup, supplied by the embedder.

    ``resolve(session)`` returns the state for that session. The
    ``versioned`` flag tells callers whether resolved states are
    kvgit-backed (``Staged``) without forcing a resolution.
    """

    versioned: bool

    def resolve(self, session: str) -> MutableMapping[str, Any]: ...


def staged_state(kv: "KVStore", *, branch: str = "main") -> "Staged":
    """Build a ``Staged`` over ``kv`` with agex's state codecs.

    The codec pair handles UnpicklableMarker wrapping and (when numpy
    is installed) chunked externalization of large array/DataFrame
    buffers — resolver-produced states must use it or chunked values
    written by agents won't round-trip. This helper is the supported
    way to get it right:

        store = Disk("/srv/agent/shared")  # one substrate...

        class BranchResolver:
            versioned = True
            def __init__(self):
                self._cache = {}
            def resolve(self, session):
                assert_safe_session(session)
                if session not in self._cache:
                    self._cache[session] = staged_state(store, branch=session)
                return self._cache[session]  # ...a branch per session

    Args:
        kv: The shared KV store.
        branch: The kvgit branch this state tracks.

    Returns:
        A ``Staged`` pinned to ``branch``, using agex's encoder/decoder.
    """
    from kvgit import Staged, VersionedKV

    from agex.state import _agex_decoder, _agex_encoder

    versioned = VersionedKV(kv, branch=branch)
    return Staged(versioned, encoder=_agex_encoder, decoder=_agex_decoder)
