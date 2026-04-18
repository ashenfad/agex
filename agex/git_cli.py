"""
Git CLI handler for termish command injection.

Translates familiar ``git`` subcommands into kvgit operations on the
agent's VFS, giving agents checkpoint-and-experiment workflows through
an interface they already know from training data.

Usage::

    from agex.git_cli import make_git_handler

    handler = make_git_handler(versioned_kv)
    execute(script, fs, commands={"git": handler})

Or register as a skill on an agent for automatic discovery::

    from agex.git_cli import register_git

    register_git(agent)

The handler is a :class:`~termish.context.CommandContext` →
:class:`~termish.context.CommandResult` | None callable suitable for
passing to termish's ``commands`` parameter.
"""

from __future__ import annotations

import difflib
from importlib import resources
from typing import TYPE_CHECKING, Any

from termish.context import CommandContext, CommandResult
from termish.errors import TerminalError

if TYPE_CHECKING:
    from kvgit import VersionedKV
    from termish.errors import CommandFunc


def register_git(agent: Any) -> None:
    """Register the git skill on an agent.

    Mounts the git usage guide at ``/skills/git/SKILL.md`` so the agent
    can discover git commands on demand via ``cat /skills/git/SKILL.md``.

    Args:
        agent: An :class:`~agex.Agent` instance.
    """
    skill_bytes = resources.files("agex.skills").joinpath("git.md").read_bytes()
    agent.skill(skill_bytes)


def make_git_handler(
    vkv: "VersionedKV",
    state: "Any | None" = None,
    vfs: "Any | None" = None,
) -> "CommandFunc":
    """Create a ``git`` command handler bound to a :class:`VersionedKV`.

    Args:
        vkv: The versioned key-value store backing the agent's VFS.
            Branch operations, commits, diffs, and history all operate
            on this instance.
        state: Optional ``Staged`` wrapper around ``vkv``.  When
            provided, ``git commit`` flushes pending staged changes
            (VFS writes) to kvgit with the commit message attached.
            Without this, ``git commit`` can only tag an empty commit.
        vfs: Optional :class:`~monkeyfs.VirtualFS` instance.  When
            provided, the handler uses the VFS's key encoding to map
            between user-facing file paths and internal kvgit keys.
            Without this, all kvgit keys are shown as-is.

    Returns:
        A termish ``CommandFunc`` that dispatches ``git`` subcommands.
    """

    def _strip(key: str) -> str:
        """Decode an internal kvgit key to a user-facing path."""
        if (
            vfs is not None
            and hasattr(vfs, "_decode_path")
            and hasattr(vfs, "_is_vfs_key")
        ):
            if vfs._is_vfs_key(key):
                try:
                    return vfs._decode_path(key)
                except Exception:
                    pass
        return key

    def _add(path: str) -> str:
        """Encode a user-facing path to an internal kvgit key."""
        if vfs is not None and hasattr(vfs, "_encode_path"):
            return vfs._encode_path(path)
        return path

    def _is_visible(key: str) -> bool:
        """Check whether a key belongs to the VFS domain (excludes metadata)."""
        if vfs is None:
            return True
        if not hasattr(vfs, "_is_vfs_key"):
            return True
        if not vfs._is_vfs_key(key):
            return False
        # Exclude VFS internal keys (metadata store, cwd, etc.)
        metadata_key = getattr(vfs, "METADATA_KEY", "__vfs_metadata__")
        cwd_key = getattr(vfs, "CWD_KEY", "__vfs_cwd__")
        return key not in (metadata_key, cwd_key)

    def handler(ctx: CommandContext) -> CommandResult | None:
        args = ctx.args
        if not args:
            ctx.stdout.write(_usage())
            return None

        subcommand = args[0]
        subargs = args[1:]

        dispatch = {
            "log": _git_log,
            "diff": _git_diff,
            "status": _git_status,
            "branch": _git_branch,
            "checkout": _git_checkout,
            "commit": _git_commit,
            "reset": _git_reset,
            "show": _git_show,
            "merge": _git_merge,
            "add": _git_add,
        }

        fn = dispatch.get(subcommand)
        if fn is None:
            raise TerminalError(f"git: '{subcommand}' is not a git command.")

        fn(
            subargs,
            ctx,
            vkv,
            _strip=_strip,
            _add=_add,
            _is_visible=_is_visible,
            _state=state,
            _vfs=vfs,
        )
        return None

    return handler


# ---------------------------------------------------------------------------
# Ref resolution
# ---------------------------------------------------------------------------


def _resolve_ref(ref: str, vkv: "VersionedKV") -> str:
    """Resolve a git-style ref (HEAD, HEAD~N, or a raw hash) to a commit hash."""
    if ref == "HEAD":
        return vkv.current_commit

    if ref.startswith("HEAD~"):
        try:
            n = int(ref[5:])
        except ValueError:
            raise TerminalError(f"git: invalid ref '{ref}'")
        commits = list(vkv.history())
        if n >= len(commits):
            raise TerminalError(
                f"git: '{ref}' is beyond the history ({len(commits)} commits)"
            )
        return commits[n]

    # Treat as a raw commit hash (prefix match)
    if len(ref) >= 7:
        for h in vkv.history():
            if h.startswith(ref):
                return h

    raise TerminalError(f"git: '{ref}' is not a valid ref")


def _short_hash(h: str) -> str:
    return h[:7]


def _is_binary(data: bytes | None) -> bool:
    """Heuristic: content is binary if it contains null bytes in the first 8KB."""
    if data is None:
        return False
    return b"\x00" in data[:8192]


def _read_file_content(
    snapshot: "Any", key: str, display_path: str, vfs: "Any | None"
) -> bytes | None:
    """Read file content from a snapshot (Staged or raw VersionedKV).

    When a ``vfs`` is provided, the snapshot is expected to be a Staged
    view whose ``.get()`` returns decoded Python values (bytes for file
    content).  Without a VFS, we read raw kvgit bytes directly.
    """
    if snapshot is None:
        return None
    val = snapshot.get(key)
    if val is None:
        return None
    if isinstance(val, bytes):
        return val
    return val


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _usage() -> str:
    return (
        "usage: git <command> [<args>]\n\n"
        "Commands:\n"
        "   log        Show commit log\n"
        "   diff       Show changes between commits\n"
        "   status     Show current branch\n"
        "   branch     List, create, or delete branches\n"
        "   checkout   Switch branches\n"
        "   commit     Record changes with a message\n"
        "   reset      Reset HEAD to a previous commit\n"
        "   show       Show file content at a commit\n"
        "   merge      Merge a branch into the current branch\n"
        "   add        (no-op) Files are tracked automatically\n"
    )


def _git_log(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    oneline = "--oneline" in args
    max_count = None
    path_filter = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-n", "--max-count"):
            i += 1
            if i < len(args):
                try:
                    max_count = int(args[i])
                except ValueError:
                    raise TerminalError(f"git log: invalid count '{args[i]}'")
        elif arg.startswith("-n") and len(arg) > 2 and arg[2:].isdigit():
            max_count = int(arg[2:])
        elif arg == "--oneline":
            pass  # already handled
        elif not arg.startswith("-") and arg != "--":
            path_filter = arg
        i += 1

    count = 0
    branch = vkv.current_branch

    for commit_hash in vkv.history():
        if max_count is not None and count >= max_count:
            break

        # If filtering by path, check if this commit touched it
        if path_filter:
            _add_fn = kw.get("_add", lambda k: k)
            internal_path = _add_fn(path_filter)
            parents = vkv.parents(commit_hash)
            if parents:
                d = vkv.diff(parents[0], commit_hash)
                touched = d.added | d.removed | d.modified
                if internal_path not in touched:
                    continue

        info = vkv.commit_info(commit_hash) or {}
        message = info.get("message", "")

        if oneline or not message:
            # Branch indicator on HEAD
            head_marker = ""
            if commit_hash == vkv.current_commit:
                head_marker = f" (HEAD -> {branch})"
            line = f"{_short_hash(commit_hash)}{head_marker}"
            if message:
                line += f" {message}"
            ctx.stdout.write(line + "\n")
        else:
            ctx.stdout.write(f"commit {commit_hash}\n")
            if commit_hash == vkv.current_commit:
                ctx.stdout.write(f"  (HEAD -> {branch})\n")
            ctx.stdout.write(f"\n    {message}\n\n")

        count += 1


def _git_diff(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    # Parse: git diff [ref_a] [ref_b] [-- path]
    refs = []
    path_filter = None
    past_separator = False

    for arg in args:
        if arg == "--":
            past_separator = True
        elif past_separator:
            path_filter = arg
        elif not arg.startswith("-"):
            refs.append(arg)

    if len(refs) == 0:
        # git diff (no args) → diff HEAD~1 vs HEAD
        commit_b = vkv.current_commit
        parents = vkv.parents(commit_b)
        if not parents:
            return  # initial commit, nothing to diff
        commit_a = parents[0]
    elif len(refs) == 1:
        commit_a = _resolve_ref(refs[0], vkv)
        commit_b = vkv.current_commit
    elif len(refs) == 2:
        commit_a = _resolve_ref(refs[0], vkv)
        commit_b = _resolve_ref(refs[1], vkv)
    else:
        raise TerminalError("git diff: too many arguments")

    _is_visible_fn = kw.get("_is_visible", lambda k: True)
    _strip_fn = kw.get("_strip", lambda k: k)
    _add_fn = kw.get("_add", lambda k: k)
    _vfs = kw.get("_vfs")

    d = vkv.diff(commit_a, commit_b)
    keys_to_diff = {k for k in (d.modified | d.added | d.removed) if _is_visible_fn(k)}
    if path_filter:
        internal_path = _add_fn(path_filter)
        keys_to_diff = {k for k in keys_to_diff if k == internal_path}

    # Use Staged checkout (decoded values) when available, otherwise raw vkv
    _staged = kw.get("_state")
    _checkout = _staged.checkout if _staged is not None else vkv.checkout
    snap_a = _checkout(commit_a)
    snap_b = _checkout(commit_b)

    for key in sorted(keys_to_diff):
        display_path = _strip_fn(key)
        old_bytes = (
            _read_file_content(snap_a, key, display_path, _vfs)
            if key not in d.added
            else None
        )
        new_bytes = (
            _read_file_content(snap_b, key, display_path, _vfs)
            if key not in d.removed
            else None
        )

        # Skip binary files — show a summary line instead of garbled diff
        if _is_binary(old_bytes) or _is_binary(new_bytes):
            ctx.stdout.write(
                f"Binary files a/{display_path} and b/{display_path} differ\n"
            )
            continue

        old_lines = (
            old_bytes.decode("utf-8").splitlines(keepends=True) if old_bytes else []
        )
        new_lines = (
            new_bytes.decode("utf-8").splitlines(keepends=True) if new_bytes else []
        )

        diff_lines = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{display_path}",
            tofile=f"b/{display_path}",
        )
        for line in diff_lines:
            ctx.stdout.write(line)
            if not line.endswith("\n"):
                ctx.stdout.write("\n")


def _git_status(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    ctx.stdout.write(f"On branch {vkv.current_branch}\n")
    ctx.stdout.write(
        "All file writes are automatically tracked. "
        "Use `git commit -m 'message'` to checkpoint.\n"
    )

    # Show recent agent-tagged commits (commits with a message)
    tagged = []
    for commit_hash in vkv.history():
        info = vkv.commit_info(commit_hash) or {}
        if info.get("message"):
            tagged.append((commit_hash, info["message"]))
        if len(tagged) >= 3:
            break

    if tagged:
        ctx.stdout.write("\nRecent commits:\n")
        for h, msg in tagged:
            ctx.stdout.write(f"  {_short_hash(h)} {msg}\n")


def _git_branch(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    if not args:
        # List branches
        branches = vkv.list_branches()
        current = vkv.current_branch
        for b in branches:
            marker = "* " if b == current else "  "
            ctx.stdout.write(f"{marker}{b}\n")
        return

    if args[0] == "-d" or args[0] == "-D":
        if len(args) < 2:
            raise TerminalError("git branch: branch name required")
        name = args[1]
        try:
            vkv.delete_branch(name)
        except ValueError as e:
            raise TerminalError(f"git branch: {e}")
        ctx.stdout.write(f"Deleted branch {name}\n")
        return

    # Create branch (don't switch)
    name = args[0]
    try:
        vkv.create_branch(name)
    except ValueError as e:
        raise TerminalError(f"git branch: {e}")
    ctx.stdout.write(f"Created branch {name}\n")


def _git_checkout(
    args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw
) -> None:
    if not args:
        raise TerminalError("git checkout: branch name required")

    if args[0] == "-b":
        if len(args) < 2:
            raise TerminalError("git checkout: branch name required after -b")
        name = args[1]
        try:
            vkv.create_branch(name)
        except ValueError as e:
            raise TerminalError(f"git checkout: {e}")
        vkv.switch_branch(name)
        ctx.stdout.write(f"Switched to a new branch '{name}'\n")
        return

    name = args[0]
    try:
        vkv.switch_branch(name)
    except ValueError as e:
        raise TerminalError(f"git checkout: {e}")
    ctx.stdout.write(f"Switched to branch '{name}'\n")


def _git_commit(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    message = None
    i = 0
    while i < len(args):
        if args[i] == "-m":
            i += 1
            if i >= len(args):
                raise TerminalError("git commit: -m requires a message")
            message = args[i]
        elif args[i].startswith("-m") and len(args[i]) > 2:
            message = args[i][2:]
        i += 1

    if not message:
        raise TerminalError(
            "git commit: please supply a message with -m 'your message'"
        )

    # If a Staged wrapper is available, flush its pending changes (VFS
    # writes from FILE/EDIT/echo) to kvgit with the message attached.
    # Otherwise, create an info-only commit on the raw VersionedKV.
    staged = kw.get("_state")
    if staged is not None and hasattr(staged, "commit"):
        result = staged.commit(info={"message": message})
    else:
        result = vkv.commit(info={"message": message})
    short = _short_hash(result.commit) if result.commit else "?"
    ctx.stdout.write(f"[{vkv.current_branch} {short}] {message}\n")


def _git_reset(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    hard = "--hard" in args
    refs = [a for a in args if not a.startswith("-")]

    if not hard:
        raise TerminalError("git reset: only --hard is supported")

    if not refs:
        raise TerminalError("git reset: need a ref (e.g. HEAD~1)")

    target = _resolve_ref(refs[0], vkv)
    if not vkv.reset_to(target):
        raise TerminalError(f"git reset: commit '{refs[0]}' not found")

    ctx.stdout.write(f"HEAD is now at {_short_hash(target)}\n")


def _git_show(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    if not args:
        raise TerminalError("git show: need a ref (e.g. HEAD:path/to/file)")

    ref_path = args[0]
    if ":" not in ref_path:
        raise TerminalError(
            "git show: use <ref>:<path> format (e.g. HEAD:helpers/utils.py)"
        )

    ref, path = ref_path.split(":", 1)
    commit_hash = _resolve_ref(ref or "HEAD", vkv)

    _staged = kw.get("_state")
    _checkout = _staged.checkout if _staged is not None else vkv.checkout
    snapshot = _checkout(commit_hash)
    if snapshot is None:
        raise TerminalError(f"git show: commit '{ref}' not found")

    _add_fn = kw.get("_add", lambda k: k)
    _vfs = kw.get("_vfs")
    internal_path = _add_fn(path)

    content = _read_file_content(snapshot, internal_path, path, _vfs)
    if content is None:
        raise TerminalError(
            f"git show: path '{path}' not found at {_short_hash(commit_hash)}"
        )

    if _is_binary(content):
        ctx.stdout.write(f"(binary file: {path}, {len(content)} bytes)\n")
    else:
        ctx.stdout.write(content.decode("utf-8", errors="replace"))


def _git_merge(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    if not args:
        raise TerminalError("git merge: branch name required")

    source_branch = args[0]
    branches = vkv.list_branches()
    if source_branch not in branches:
        raise TerminalError(f"git merge: branch '{source_branch}' not found")

    if source_branch == vkv.current_branch:
        raise TerminalError("git merge: cannot merge a branch into itself")

    # Open a read-only view of the source branch at its HEAD.
    from kvgit.versioned.kv import VersionedKV as _VKV

    source_vkv = _VKV(vkv.store, branch=source_branch)
    source_head = source_vkv.current_commit

    # Compute diff between current and source
    d = vkv.diff(vkv.current_commit, source_head)
    if not d.added and not d.modified and not d.removed:
        ctx.stdout.write("Already up to date.\n")
        return

    # Build updates from source
    updates: dict[str, bytes] = {}
    removals: set[str] = set()
    for key in d.added | d.modified:
        val = source_vkv.get(key)
        if val is not None:
            updates[key] = val
    for key in d.removed:
        removals.add(key)

    try:
        result = vkv.commit(
            updates=updates,
            removals=removals,
            info={"message": f"Merge branch '{source_branch}'"},
        )
    except Exception as e:
        raise TerminalError(f"git merge: conflict — {e}")

    short = _short_hash(result.commit) if result.commit else "?"
    ctx.stdout.write(f"Merge made: {short} Merge branch '{source_branch}'\n")
    if d.added:
        ctx.stdout.write(f"  {len(d.added)} file(s) added\n")
    if d.modified:
        ctx.stdout.write(f"  {len(d.modified)} file(s) modified\n")
    if d.removed:
        ctx.stdout.write(f"  {len(d.removed)} file(s) removed\n")


def _git_add(args: list[str], ctx: CommandContext, vkv: "VersionedKV", **kw) -> None:
    # No-op — files are tracked automatically in the VFS.
    pass
