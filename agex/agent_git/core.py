"""VirtualGit — agent-view git core.

Owns the agent's virtual git semantics (branches, index, log) and is
the single consumer of the kvgit / Staged / VFS substrate.  The CLI in
:mod:`agex.agent_git.cli` is a thin parser/formatter around this
class; tests and Python helpers can drive it directly.

This module deliberately speaks only in agent terms.  Branch operations
never touch real kvgit branches — those would move framework state
(event log, REPL namespace, agent memory) the agent should not see.
File-content moves go through ``Staged`` writes (the same pattern the
prior ``git reset`` already used) so the next ``commit_state`` carries
them forward as a forward commit, leaving kvgit HEAD on its real path.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .metadata import METADATA_KEY, Metadata
from .refs import (
    InvalidRef,
    all_ancestors,
    merge_base,
    resolve_ref,
    virtual_parents,
    walk_virtual_ancestry,
)

if TYPE_CHECKING:
    from kvgit import Staged, VersionedKV


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AgentGitError(ValueError):
    """Base class for agent-git operation errors raised by VirtualGit.

    The CLI in :mod:`agex.agent_git.cli` translates these into termish
    :class:`~termish.errors.TerminalError` instances; direct callers
    can catch :class:`AgentGitError` (or specific subclasses) to react
    programmatically.
    """


class BranchExists(AgentGitError):
    pass


class BranchNotFound(AgentGitError):
    pass


class UnbornBranch(AgentGitError):
    """Operation requires at least one commit on the current branch."""


class PendingChanges(AgentGitError):
    """Refused because the working tree has uncommitted visible changes."""


class NothingToCommit(AgentGitError):
    pass


class PathSpecError(AgentGitError):
    """A user-supplied path didn't match any known file."""


class BranchNotMerged(AgentGitError):
    """Refused to delete a branch whose tip isn't reachable from HEAD."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class AgentCommit:
    """An agent-driven commit, as the agent sees it."""

    hash: str
    message: str
    virtual_branch: str | None
    virtual_parents: list[str]
    files: list[str] | None  # decoded user paths, or None when not annotated

    @property
    def short_hash(self) -> str:
        return self.hash[:7]


@dataclass
class Status:
    """Working-tree status against the current virtual branch."""

    branch: str
    staged: list[str]  # decoded user paths, sorted
    unstaged: list[str]

    @property
    def is_clean(self) -> bool:
        return not (self.staged or self.unstaged)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_binary(data: bytes | None) -> bool:
    """Heuristic: content is binary if it contains a NUL in the first 8KB."""
    if data is None:
        return False
    return b"\x00" in data[:8192]


# ---------------------------------------------------------------------------
# VirtualGit
# ---------------------------------------------------------------------------


class VirtualGit:
    """Agent-view git operations over a kvgit / Staged / VFS substrate.

    Args:
        vkv: The :class:`kvgit.VersionedKV` backing the agent's session.
            Used for commit-info reads, history walks, and hash-level
            diffs.  Branch APIs on this object are NEVER called.
        state: The :class:`kvgit.Staged` wrapping ``vkv``.  All file
            reads and writes go through it; this is also where the
            agent-git metadata blob lives.
        vfs: Optional :class:`monkeyfs.VirtualFS`.  When provided,
            user-facing paths are translated to/from internal kvgit
            keys via the VFS's encode/decode helpers.  Without a VFS,
            kvgit keys are treated as user paths verbatim — useful for
            tests that bypass monkeyfs.
    """

    def __init__(
        self,
        vkv: "VersionedKV",
        state: "Staged",
        vfs: Any | None = None,
    ) -> None:
        self._vkv = vkv
        self._state = state
        self._vfs = vfs

    # -- Substrate helpers ---------------------------------------------------

    def _load_metadata(self) -> Metadata:
        return Metadata.load(self._state)

    def _decode(self, key: str) -> str:
        """Internal kvgit key → user-facing path."""
        if (
            self._vfs is not None
            and hasattr(self._vfs, "_decode_path")
            and hasattr(self._vfs, "_is_vfs_key")
            and self._vfs._is_vfs_key(key)
        ):
            try:
                return self._vfs._decode_path(key)
            except Exception:
                pass
        return key

    def _encode(self, path: str) -> str:
        """User-facing path → internal kvgit key."""
        if self._vfs is not None and hasattr(self._vfs, "_encode_path"):
            return self._vfs._encode_path(path)
        return path

    def _is_visible(self, key: str) -> bool:
        """Whether a key is a user-file VFS key (excludes all metadata)."""
        if key == METADATA_KEY:
            return False
        if self._vfs is None:
            return True
        if not hasattr(self._vfs, "_is_vfs_key"):
            return True
        if not self._vfs._is_vfs_key(key):
            return False
        # Exclude monkeyfs's own internal keys (manifest, cwd).
        vfs_metadata_key = getattr(self._vfs, "METADATA_KEY", "__vfs_metadata__")
        cwd_key = getattr(self._vfs, "CWD_KEY", "__vfs_cwd__")
        return key not in (vfs_metadata_key, cwd_key)

    # -- Branch state -------------------------------------------------------

    @property
    def current_branch(self) -> str:
        return self._load_metadata().current

    def list_branches(self) -> list[str]:
        return sorted(self._load_metadata().branches.keys())

    def head(self) -> str | None:
        """Commit hash of the current branch's tip, or None if unborn."""
        return self._load_metadata().head

    # -- Ref resolution -----------------------------------------------------

    def resolve_ref(self, ref: str) -> str:
        return resolve_ref(ref, self._vkv, self._load_metadata())

    # -- Status -------------------------------------------------------------

    def status(self) -> Status:
        meta = self._load_metadata()
        modified = self._modified_keys(meta)
        staged = sorted(modified & meta.index)
        unstaged = sorted(modified - meta.index)
        return Status(
            branch=meta.current,
            staged=[self._decode(k) for k in staged],
            unstaged=[self._decode(k) for k in unstaged],
        )

    def _modified_keys(self, meta: Metadata) -> set[str]:
        """Visible keys whose live content differs from the branch tip.

        On an unborn branch every visible working-tree key counts as
        modified — there is no baseline to compare against.
        """
        head = meta.head
        if head is None:
            return {k for k in self._state.keys() if self._is_visible(k)}
        return self._diff_keyset(head, None)

    # -- Log ----------------------------------------------------------------

    def log(
        self,
        *,
        max_count: int | None = None,
        path: str | None = None,
    ) -> list[AgentCommit]:
        """Walk virtual ancestry from the current branch's tip.

        Optional ``path`` filters to commits that touched the file.
        Path-touch detection diffs against the commit's first *virtual*
        parent — the previous tagged-agent commit on the same branch,
        which matches what an agent reading their own log expects.
        Falls back to skipping the commit when no virtual parent is
        recorded (root commit on a branch).
        """
        meta = self._load_metadata()
        head = meta.head
        if head is None:
            return []

        path_key = self._encode(path) if path else None

        out: list[AgentCommit] = []
        for h in walk_virtual_ancestry(self._vkv, head):
            if path_key is not None:
                v_parents = virtual_parents(self._vkv, h)
                if v_parents:
                    d = self._vkv.diff(v_parents[0], h)
                    if path_key not in (d.added | d.removed | d.modified):
                        continue
                else:
                    # Root agent commit (no virtual parent).  Real git
                    # includes the initial commit in ``git log -- path``
                    # when that commit introduced the file; we mirror
                    # that by checking presence at the commit itself.
                    snap = self._state.checkout(h)
                    if snap is None or path_key not in snap:
                        continue

            out.append(self._make_commit(h))
            if max_count is not None and len(out) >= max_count:
                break
        return out

    def _make_commit(self, commit_hash: str) -> AgentCommit:
        info = self._vkv.commit_info(commit_hash) or {}
        files = info.get("files")
        return AgentCommit(
            hash=commit_hash,
            message=info.get("message", ""),
            virtual_branch=info.get("virtual_branch"),
            virtual_parents=list(info.get("virtual_parents") or []),
            files=list(files) if files is not None else None,
        )

    # -- Show ---------------------------------------------------------------

    def show(self, commit_hash: str, path: str) -> bytes:
        """Read file content at a specific commit.

        Raises :class:`InvalidRef` if the commit doesn't exist and
        :class:`FileNotFoundError` if the path isn't present at the
        commit (or isn't a regular file).
        """
        snap = self._state.checkout(commit_hash)
        if snap is None:
            raise InvalidRef(f"commit '{commit_hash[:7]}' not found")
        key = self._encode(path)
        val = snap.get(key)
        if val is None:
            raise FileNotFoundError(f"path '{path}' not found at {commit_hash[:7]}")
        if not isinstance(val, bytes):
            # Non-bytes here means we resolved a non-VFS key (REPL var,
            # event log, etc.) — never a file from the agent's view.
            raise FileNotFoundError(f"path '{path}' is not a file at {commit_hash[:7]}")
        return val

    # -- Diff ---------------------------------------------------------------

    def diff(
        self,
        a: str | None = None,
        b: str | None = None,
        *,
        path: str | None = None,
    ) -> str:
        """Unified diff between two views.

        ``a`` / ``b`` are commit hashes (already resolved by the caller
        via :meth:`resolve_ref`) or ``None`` meaning "the live working
        view".  Default (both ``None``) diffs HEAD vs working — matching
        real git's plain ``git diff``.
        """
        meta = self._load_metadata()

        if a is None and b is None:
            head = meta.head
            if head is None:
                return ""
            a = head

        keys = self._diff_keyset(a, b)
        if path:
            target = self._encode(path)
            keys = {k for k in keys if k == target}

        snap_a = self._state.checkout(a) if a is not None else self._state
        snap_b = self._state.checkout(b) if b is not None else self._state
        if snap_a is None or snap_b is None:
            raise InvalidRef("commit not found")

        return self._render_diff(snap_a, snap_b, keys)

    def _diff_keyset(self, a: str | None, b: str | None) -> set[str]:
        """Visible keys that differ between two views.

        When both sides are commits, uses kvgit's hash-level
        :meth:`VersionedKV.diff` (HAMT root comparison — O(log N)).
        When either side is the live working view, falls back to
        content comparison since there's no commit to compare hashes
        against.
        """
        if a is not None and b is not None:
            d = self._vkv.diff(a, b)
            return {
                k for k in (d.added | d.modified | d.removed) if self._is_visible(k)
            }

        snap_a = self._state.checkout(a) if a is not None else self._state
        snap_b = self._state.checkout(b) if b is not None else self._state
        if snap_a is None or snap_b is None:
            raise InvalidRef("commit not found")

        a_keys = {k for k in snap_a.keys() if self._is_visible(k)}
        b_keys = {k for k in snap_b.keys() if self._is_visible(k)}
        result = a_keys.symmetric_difference(b_keys)
        for k in a_keys & b_keys:
            if snap_a.get(k) != snap_b.get(k):
                result.add(k)
        return result

    def _render_diff(self, snap_a, snap_b, keys: set[str]) -> str:
        out: list[str] = []
        for key in sorted(keys):
            display = self._decode(key)
            old = snap_a.get(key)
            new = snap_b.get(key)
            old_bytes = old if isinstance(old, bytes) else None
            new_bytes = new if isinstance(new, bytes) else None

            if is_binary(old_bytes) or is_binary(new_bytes):
                out.append(f"Binary files a/{display} and b/{display} differ\n")
                continue

            # ``errors="replace"`` keeps ``git diff`` from crashing on
            # files that aren't strictly UTF-8 (latin-1, mojibake,
            # partially-corrupted text).  The is_binary check above
            # filters NUL-containing files; everything else is
            # rendered with ``�`` substitutions for invalid bytes.
            old_lines = (
                old_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
                if old_bytes
                else []
            )
            new_lines = (
                new_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
                if new_bytes
                else []
            )
            for line in difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{display}",
                tofile=f"b/{display}",
            ):
                if not line.endswith("\n"):
                    line += "\n"
                out.append(line)

        return "".join(out)

    # -- Working-tree application -------------------------------------------

    def _flush_alignment(self) -> None:
        """Persist pending Staged changes as an unmessaged kvgit commit.

        Used after checkout / reset / fast-forward merge to bring the
        kvgit physical chain in line with virtual semantics *before*
        the next agent commit.  Without this, a subsequent selective
        ``state.commit(keys=...)`` would inherit unrelated keys from
        the physical parent (e.g., a file removed by checkout would
        re-appear in the next commit because selective flush ignores
        ``_removals`` outside its key set).

        Equivalent to what ``commit_state`` does at turn boundaries —
        just done synchronously so two virtual operations in the same
        terminal_action stay self-consistent.
        """
        if self._state.has_changes:
            self._state.commit(info=None)

    def _apply_file_view(self, target_hash: str) -> None:
        """Make the live working view match ``target_hash`` (visible keys only).

        Writes through ``Staged`` so the next ``commit_state`` carries
        the change forward as a forward kvgit commit, leaving real
        kvgit HEAD on its native chain.  Non-VFS keys (event log, REPL
        namespace, agent memory) are *not* touched.
        """
        target = self._state.checkout(target_hash)
        if target is None:
            raise InvalidRef(f"commit '{target_hash[:7]}' not found")

        cur_keys = {k for k in self._state.keys() if self._is_visible(k)}
        target_keys = {k for k in target.keys() if self._is_visible(k)}

        for key in cur_keys - target_keys:
            try:
                del self._state[key]
            except KeyError:
                pass

        for key in target_keys:
            target_val = target.get(key)
            if target_val is None:
                continue
            cur_val = self._state.get(key) if key in cur_keys else None
            if cur_val != target_val:
                self._state[key] = target_val

    # -- add / rm -----------------------------------------------------------

    def add(self, paths: list[str]) -> None:
        """Stage paths for the next commit.

        ``["."]`` or ``["-A"]`` stages every currently-modified file.
        Non-existent / unmodified paths raise :class:`PathSpecError`,
        matching real git's ``pathspec`` behaviour.
        """
        if not paths:
            raise PathSpecError("nothing specified")

        meta = self._load_metadata()
        modified = self._modified_keys(meta)

        if paths == ["."] or paths == ["-A"]:
            meta.index.update(modified)
            meta.save(self._state)
            return

        # Build the universe of "known" keys: working tree + branch tip.
        # Either is sufficient justification to ``add`` a path.
        known = {k for k in self._state.keys() if self._is_visible(k)}
        if meta.head is not None:
            head_snap = self._state.checkout(meta.head)
            if head_snap is not None:
                known.update(k for k in head_snap.keys() if self._is_visible(k))

        for path in paths:
            key = self._encode(path)
            if key not in known and key not in modified:
                raise PathSpecError(f"pathspec '{path}' did not match any files")
            meta.index.add(key)
        meta.save(self._state)

    def rm(self, paths: list[str], *, recursive: bool = False) -> None:
        """Remove paths from the working tree and stage the deletion.

        Returns silently on success.  With ``recursive=True``, removes
        every visible key whose path starts with the given prefix.
        """
        if not paths:
            raise PathSpecError("nothing specified")

        meta = self._load_metadata()
        removed: list[str] = []

        for path in paths:
            internal = self._encode(path)

            if recursive:
                # Match anything in the working tree under this prefix.
                # When the user typed a directory, the encoded "key" of
                # ``foo/`` is the encoding of the path "foo/", not a
                # prefix of all keys under it — so we also try
                # encoding ``foo/`` and use that as the prefix.
                candidates: list[str] = []
                for key in list(self._state.keys()):
                    if not self._is_visible(key):
                        continue
                    decoded = self._decode(key)
                    if decoded == path or decoded.startswith(path.rstrip("/") + "/"):
                        candidates.append(key)
                if not candidates:
                    raise PathSpecError(f"pathspec '{path}' did not match any files")
                for key in candidates:
                    try:
                        del self._state[key]
                        removed.append(self._decode(key))
                        meta.index.add(key)
                    except KeyError:
                        pass
            else:
                if internal in self._state:
                    try:
                        del self._state[internal]
                        removed.append(path)
                        meta.index.add(internal)
                    except KeyError:  # pragma: no cover — race-only
                        raise PathSpecError(
                            f"pathspec '{path}' did not match any files"
                        )
                else:
                    # File isn't in the working tree.  Real git still
                    # accepts ``git rm`` on a path tracked at HEAD that
                    # was already deleted from the workspace —
                    # idempotently re-stages the deletion.  Without
                    # this, an agent can't ``git rm`` a file they
                    # already removed via shell ``rm`` or a Python
                    # script.
                    in_head = False
                    if meta.head is not None:
                        head_snap = self._state.checkout(meta.head)
                        if head_snap is not None:
                            in_head = internal in head_snap
                    if not in_head:
                        raise PathSpecError(
                            f"pathspec '{path}' did not match any files"
                        )
                    meta.index.add(internal)
                    removed.append(path)

        meta.save(self._state)

    # -- commit -------------------------------------------------------------

    def commit(self, message: str) -> AgentCommit:
        """Record a new agent commit on the current branch.

        Selective when :attr:`Metadata.index` is non-empty (only those
        keys are flushed); full when the index is empty.  Modified
        content is re-staged into the ``Staged`` buffer so a selective
        flush picks up files that ``commit_state`` already pushed
        through to kvgit between turns.

        Updates the branch ref in metadata and clears the index on
        success.  The metadata write is staged for the next
        ``commit_state`` to persist — so a hard crash between this
        call and the next turn boundary leaves an orphan kvgit commit
        unreachable from any virtual branch (acceptable; the agent's
        view re-converges on retry).
        """
        meta = self._load_metadata()
        modified = self._modified_keys(meta)
        if not modified:
            raise NothingToCommit("nothing to commit, working tree clean")

        if meta.index:
            keys_to_commit = modified & meta.index
            if not keys_to_commit:
                raise NothingToCommit(
                    "nothing to commit (staged files match the branch tip)"
                )
        else:
            keys_to_commit = set(modified)

        # Re-stage current values so a selective flush picks them up
        # even when ``commit_state`` already pushed them through to
        # kvgit between turns.  Deletions that landed in kvgit remain
        # absent from the new commit naturally (parented to the latest
        # kvgit HEAD which already excludes them).
        for key in keys_to_commit:
            cur_val = self._state.get(key)
            if cur_val is not None:
                self._state[key] = cur_val
            elif key in self._state:
                # Key still present in some buffered form; force removal
                # so the selective commit flushes the deletion.
                try:
                    del self._state[key]
                except KeyError:
                    pass

        info = {
            "message": message,
            "files": sorted(self._decode(k) for k in keys_to_commit),
            "virtual_branch": meta.current,
            "virtual_parents": [meta.head] if meta.head else [],
        }
        result = self._state.commit(keys=keys_to_commit, info=info)
        new_hash = result.commit
        if new_hash is None:
            raise AgentGitError("commit was abandoned (conflict)")

        meta.branches[meta.current] = new_hash
        meta.index.clear()
        meta.save(self._state)

        return self._make_commit(new_hash)

    # -- reset --------------------------------------------------------------

    def reset(self, target: str, *, hard: bool = True) -> None:
        """Restore the working tree to ``target`` and rewind the branch ref.

        ``target`` is a commit hash already resolved by the caller.
        Only ``hard`` is supported.  This is a *virtual* reset: kvgit
        HEAD is not moved.  The branch ref in metadata is rewound to
        ``target`` so subsequent ``git log`` / ``HEAD~N`` reflect the
        reset, matching real git's ``reset --hard`` behaviour.
        """
        if not hard:
            raise AgentGitError("only --hard is supported")

        meta = self._load_metadata()
        self._apply_file_view(target)

        meta.branches[meta.current] = target
        meta.index.clear()
        meta.save(self._state)
        self._flush_alignment()

    # -- branch operations -------------------------------------------------

    def create_branch(self, name: str) -> None:
        """Create a new virtual branch pointing at the current branch's tip.

        Raises :class:`BranchExists` if ``name`` is already a branch
        and :class:`UnbornBranch` if the current branch has no
        commits (mirroring real git's ``Not a valid object name``).
        """
        if not name:
            raise AgentGitError("branch name required")

        meta = self._load_metadata()
        if name in meta.branches:
            raise BranchExists(f"branch '{name}' already exists")
        if meta.head is None:
            raise UnbornBranch(
                f"cannot create branch '{name}': '{meta.current}' has no commits yet"
            )
        meta.branches[name] = meta.head
        meta.save(self._state)

    def delete_branch(self, name: str, *, force: bool = False) -> None:
        """Delete a virtual branch.

        Without ``force``, refuses to delete a branch whose tip isn't
        reachable from the current branch (i.e., would lose commits).
        """
        meta = self._load_metadata()
        if name not in meta.branches:
            raise BranchNotFound(f"branch '{name}' not found")
        if name == meta.current:
            raise AgentGitError(f"cannot delete branch '{name}' currently checked out")

        if not force:
            tip = meta.branches[name]
            reachable = all_ancestors(self._vkv, meta.head)
            if tip not in reachable:
                raise BranchNotMerged(
                    f"branch '{name}' is not fully merged.\n"
                    f"Use force-delete to discard its commits."
                )

        del meta.branches[name]
        meta.save(self._state)

    def checkout(
        self,
        name: str,
        *,
        create: bool = False,
        force: bool = False,
    ) -> None:
        """Switch the current virtual branch.

        ``create=True`` creates the branch first (like ``git checkout -b``).
        Without ``force``, refuses if the working tree has visible
        modifications relative to the current branch tip — that's the
        equivalent of real git's "would be overwritten by checkout"
        guard, but content-based instead of buffer-based so it catches
        edits that ``commit_state`` already flushed through kvgit.

        On success the working tree is rewritten to match the target
        branch (visible keys only — non-VFS state is untouched), the
        branch ref in metadata advances, and the index is cleared.
        """
        if not name:
            raise AgentGitError("branch name required")

        meta = self._load_metadata()

        if create:
            if name in meta.branches:
                raise BranchExists(f"branch '{name}' already exists")
            if meta.head is None:
                raise UnbornBranch(
                    f"cannot create branch '{name}': "
                    f"'{meta.current}' has no commits yet"
                )
            meta.branches[name] = meta.head

        if name not in meta.branches:
            raise BranchNotFound(f"branch '{name}' does not exist")

        if name == meta.current and not create:
            return  # no-op

        if not force:
            modified = self._modified_keys(meta)
            if modified:
                raise PendingChanges(
                    "your local changes would be lost.\n"
                    "Please commit your changes (git commit -m '...') "
                    "before switching branches."
                )

        target = meta.branches[name]
        self._apply_file_view(target)

        meta.current = name
        meta.index.clear()
        meta.save(self._state)
        self._flush_alignment()

    # -- merge --------------------------------------------------------------

    def merge(self, source: str, *, force: bool = False) -> AgentCommit | None:
        """Merge ``source`` virtual branch into the current branch.

        Returns:
            The merge :class:`AgentCommit` for a true merge, the
            fast-forward target for a fast-forward, or ``None`` when
            already up to date.

        Semantics (v1):

        * ``source`` is reachable from current → already up to date.
        * Current is reachable from source → fast-forward (no merge
          commit; branch ref just advances to ``source``).
        * Otherwise → "source wins" merge.  Files differing between
          the two tips take ``source``'s value; files unique to current
          are kept; files unique to source are added; files removed
          on source are removed.  No three-way merge is attempted.
        """
        if not source:
            raise AgentGitError("branch name required")

        meta = self._load_metadata()
        if source == meta.current:
            raise AgentGitError("cannot merge a branch into itself")
        if source not in meta.branches:
            raise BranchNotFound(f"branch '{source}' not found")

        source_tip = meta.branches[source]
        current_tip = meta.head
        if current_tip is None:
            raise UnbornBranch(
                f"current branch '{meta.current}' has no commits to merge into"
            )

        if source_tip == current_tip:
            return None  # already up to date

        # If source is an ancestor of current, current already has it.
        if source_tip in all_ancestors(self._vkv, current_tip):
            return None

        if not force:
            if self._modified_keys(meta):
                raise PendingChanges(
                    "your local changes would be overwritten.\n"
                    "Please commit your changes (git commit -m '...') "
                    "before merging."
                )

        # Fast-forward when current is in source's ancestry.
        if current_tip in all_ancestors(self._vkv, source_tip):
            self._apply_file_view(source_tip)
            meta.branches[meta.current] = source_tip
            meta.index.clear()
            meta.save(self._state)
            self._flush_alignment()
            return self._make_commit(source_tip)

        # True merge: apply only the changes ``source`` made *since
        # the merge base*.  Files current changed independently of
        # source are left alone; files both branches changed (a real
        # conflict in 3-way merge terms) take source's value — v1's
        # "theirs wins on conflict" approximation.
        base = merge_base(self._vkv, current_tip, source_tip)
        if base is None:
            raise AgentGitError(
                "merge: no common ancestor (refusing to merge unrelated histories)"
            )
        diff = self._vkv.diff(base, source_tip)
        affected = {
            k
            for k in (diff.added | diff.modified | diff.removed)
            if self._is_visible(k)
        }
        source_snap = self._state.checkout(source_tip)
        if source_snap is None:
            raise AgentGitError(f"merge: source commit '{source_tip[:7]}' not found")

        for key in affected:
            if key in diff.removed:
                if key in self._state:
                    try:
                        del self._state[key]
                    except KeyError:
                        pass
            else:
                val = source_snap.get(key)
                if val is not None:
                    self._state[key] = val

        info = {
            "message": f"Merge branch '{source}'",
            "files": sorted(self._decode(k) for k in affected),
            "virtual_branch": meta.current,
            "virtual_parents": [current_tip, source_tip],
        }
        result = self._state.commit(keys=affected, info=info)
        new_hash = result.commit
        if new_hash is None:
            raise AgentGitError("merge: commit was abandoned (conflict)")

        meta.branches[meta.current] = new_hash
        meta.index.clear()
        meta.save(self._state)

        return self._make_commit(new_hash)
