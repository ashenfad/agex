"""Ref resolution and virtual ancestry walks.

Agent commits record their *virtual* parent(s) in ``commit_info``
(``virtual_parents``) — these point to the previous tip of the same
virtual branch (one parent for a normal commit, two for a merge).
That graph is independent of kvgit's physical commit chain, which
includes framework commits made by ``commit_state`` at turn boundaries
and is therefore not what the agent should walk for ``git log`` /
``HEAD~N`` / merge-base.

This module provides the navigation primitives over that virtual graph.
It does not depend on termish or any CLI plumbing — the CLI translates
:class:`InvalidRef` into a :class:`~termish.errors.TerminalError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from kvgit import VersionedKV

    from .metadata import Metadata


# Minimum hash prefix length accepted by :func:`resolve_ref`.  Matches
# common git tooling and the prior behaviour of agex's git CLI.
HASH_PREFIX_MIN_LEN = 7


class InvalidRef(ValueError):
    """Raised when a ref string cannot be resolved to a commit hash."""


# ---------------------------------------------------------------------------
# Agent-commit identification
# ---------------------------------------------------------------------------


def is_agent_commit(vkv: "VersionedKV", commit_hash: str) -> bool:
    """Whether ``commit_hash`` was created by an explicit ``git commit -m``.

    Framework commits (made by ``commit_state`` at turn boundaries) carry no
    ``message`` in their info dict; agent-driven commits always do.
    """
    info = vkv.commit_info(commit_hash)
    return bool(info and info.get("message"))


def all_agent_commits(vkv: "VersionedKV") -> list[str]:
    """All agent-driven commits across the kvgit store, newest-first.

    Used as the search space for hash-prefix resolution and for sanity
    checks; *not* used for ``git log`` output, which walks per-branch
    virtual ancestry instead.
    """
    return [h for h in vkv.history() if is_agent_commit(vkv, h)]


# ---------------------------------------------------------------------------
# Virtual ancestry
# ---------------------------------------------------------------------------


def virtual_parents(vkv: "VersionedKV", commit_hash: str) -> list[str]:
    """Virtual parents recorded in ``commit_info`` for ``commit_hash``.

    Returns the empty list if the commit has no recorded virtual
    parents (root commit on a branch, or a commit predating the
    virtual-branch system).
    """
    info = vkv.commit_info(commit_hash) or {}
    parents = info.get("virtual_parents")
    if not parents:
        return []
    # commit_info round-trips through JSON in some kvgit backends, so
    # the list may be a tuple — normalise.
    return list(parents)


def merge_base(vkv: "VersionedKV", a: str | None, b: str | None) -> str | None:
    """Lowest common ancestor of two commits in the virtual DAG.

    Returns ``None`` when either input is ``None`` or the two
    histories share no ancestor (unrelated trees).  When ``a == b`` or
    one is reachable from the other, returns the deeper of the two
    (matching real git's ``git merge-base`` behaviour).
    """
    if a is None or b is None:
        return None
    if a == b:
        return a

    a_ancestors = all_ancestors(vkv, a)
    if b in a_ancestors:
        return b

    # BFS from b — the first ancestor we hit that's in a_ancestors is
    # the LCA.  BFS guarantees the *closest* common ancestor first.
    from collections import deque

    seen: set[str] = set()
    queue = deque([b])
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in a_ancestors:
            return cur
        queue.extend(virtual_parents(vkv, cur))
    return None


def all_ancestors(vkv: "VersionedKV", head: str | None) -> set[str]:
    """All virtual ancestors of ``head`` reachable via either parent.

    Unlike :func:`walk_virtual_ancestry` this is a full DAG walk:
    merge commits expose *both* parents, so the result is the closure
    needed to answer "is X reachable from HEAD?" — used by
    ``branch -d`` to decide whether a branch is fully merged.

    Returns the empty set for an unborn branch.  Includes ``head``
    itself in the result.
    """
    if head is None:
        return set()
    seen: set[str] = set()
    stack: list[str] = [head]
    while stack:
        h = stack.pop()
        if h in seen:
            continue
        seen.add(h)
        stack.extend(virtual_parents(vkv, h))
    return seen


def walk_virtual_ancestry(vkv: "VersionedKV", head: str | None) -> Iterator[str]:
    """Yield commits along the first-parent virtual ancestry from ``head``.

    Linear walk via ``virtual_parents[0]``; for merge commits this
    follows the "into" branch (the branch the merge was made on),
    matching real git's first-parent log convention.

    Yields nothing when ``head`` is ``None`` (unborn branch).
    Defensive against pathological cycles via a visited-set guard;
    cycles shouldn't be reachable through content-addressed commits
    but a corrupt store shouldn't deadlock the CLI.
    """
    if head is None:
        return
    seen: set[str] = set()
    cur: str | None = head
    while cur is not None and cur not in seen:
        seen.add(cur)
        yield cur
        parents = virtual_parents(vkv, cur)
        cur = parents[0] if parents else None


# ---------------------------------------------------------------------------
# Ref resolution
# ---------------------------------------------------------------------------


def resolve_ref(
    ref: str,
    vkv: "VersionedKV",
    metadata: "Metadata",
) -> str:
    """Resolve an agent-supplied ref string to a commit hash.

    Resolution order:

    1. ``HEAD``: the tip of :attr:`Metadata.current`.
    2. ``HEAD~N`` (``N >= 0``): walk N steps back through virtual
       ancestry from ``HEAD``.
    3. Branch name: lookup in :attr:`Metadata.branches`.
    4. Hash prefix (>= 7 chars): match against any agent-tagged commit.

    Raises :class:`InvalidRef` for empty input, unborn ``HEAD``,
    ``HEAD~N`` exceeding ancestry length, unknown branch names, or
    unmatched / ambiguous hash prefixes.
    """
    if not ref:
        raise InvalidRef("empty ref")

    if ref == "HEAD":
        head = metadata.head
        if head is None:
            raise InvalidRef(
                f"HEAD is unborn (branch '{metadata.current}' has no commits)"
            )
        return head

    if ref.startswith("HEAD~"):
        try:
            n = int(ref[len("HEAD~") :])
        except ValueError:
            raise InvalidRef(f"invalid ref '{ref}'")
        if n < 0:
            raise InvalidRef(f"invalid ref '{ref}'")
        ancestry = list(walk_virtual_ancestry(vkv, metadata.head))
        if not ancestry:
            raise InvalidRef(
                f"HEAD is unborn (branch '{metadata.current}' has no commits)"
            )
        if n >= len(ancestry):
            raise InvalidRef(
                f"'{ref}' is beyond the history "
                f"({len(ancestry)} commit{'s' if len(ancestry) != 1 else ''} "
                f"on branch '{metadata.current}')"
            )
        return ancestry[n]

    # Branch names take precedence over hash prefixes — matches real git.
    if ref in metadata.branches:
        return metadata.branches[ref]

    if len(ref) >= HASH_PREFIX_MIN_LEN:
        matches = [h for h in all_agent_commits(vkv) if h.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise InvalidRef(f"ambiguous ref '{ref}' matches {len(matches)} commits")

    raise InvalidRef(f"'{ref}' is not a valid ref")
