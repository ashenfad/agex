"""Schema and I/O for the agent-view git metadata blob.

Holds the agent's "virtual" git state — current branch, branch refs,
staged-file index — in a single value at a reserved key in the kvgit
store.  Deliberately separate from kvgit's own branch state: real
kvgit branches own the entire keyspace (event log, REPL, VFS) and
must never be moved by an agent ``git`` command, so the agent layer
keeps its own bookkeeping here.

The blob is just a plain dict round-tripped through whatever encoder
the surrounding ``Staged`` is configured with (pickle by default, the
chunked codec in production agex setups).  It does not start with
monkeyfs's ``__vfs_`` prefix, so :class:`monkeyfs.VirtualFS` does not
treat it as a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Reserved kvgit key.  Plain dict value, encoded via the surrounding
# Staged's encoder.  The leading/trailing dunders are stylistic — the
# isolation contract is "must not collide with any VFS-encoded path",
# which holds because monkeyfs only treats keys with the ``__vfs_``
# prefix as files.
METADATA_KEY = "__agex_git__"

DEFAULT_BRANCH = "main"


@dataclass
class Metadata:
    """Agent-view git state.

    Attributes:
        current: Name of the currently checked-out virtual branch.
            Always set; defaults to ``"main"``.  May refer to a branch
            that has no entry in :attr:`branches` yet — that's the
            "unborn" state of a fresh store before the first commit,
            mirroring real ``git init``.
        branches: Mapping of branch name → kvgit commit hash.  An entry
            exists for every virtual branch that has at least one
            commit.  Empty for a fresh store.
        index: Internal kvgit keys (encoded VFS keys) that the agent
            has explicitly staged via ``git add``.  The next
            ``git commit`` flushes only these keys when the set is
            non-empty.
    """

    current: str = DEFAULT_BRANCH
    branches: dict[str, str] = field(default_factory=dict)
    index: set[str] = field(default_factory=set)

    @property
    def head(self) -> str | None:
        """Commit hash for :attr:`current`, or ``None`` if unborn."""
        return self.branches.get(self.current)

    @classmethod
    def load(cls, state: Any) -> "Metadata":
        """Read metadata from ``state``.  Returns defaults if absent.

        Args:
            state: A ``Staged`` (or compatible mapping) over the
                underlying kvgit store.
        """
        raw = state.get(METADATA_KEY)
        if raw is None:
            return cls()
        # Tolerant of partial / older blobs so a load can never crash
        # the git CLI on a degraded store.
        return cls(
            current=raw.get("current", DEFAULT_BRANCH),
            branches=dict(raw.get("branches") or {}),
            index=set(raw.get("index") or ()),
        )

    def save(self, state: Any) -> None:
        """Write metadata back to ``state`` as a fresh blob."""
        state[METADATA_KEY] = {
            "current": self.current,
            "branches": dict(self.branches),
            # Sorted list for stable serialisation; load() coerces back to set.
            "index": sorted(self.index),
        }
