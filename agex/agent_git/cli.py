"""Termish adapter for :class:`agex.agent_git.VirtualGit`.

Translates ``git <subcommand>`` from a termish ``CommandContext``
into method calls on a ``VirtualGit`` instance and formats the result
back to ``stdout``.  Argument parsing, error translation, and output
formatting live here; semantics live in :mod:`agex.agent_git.core`.

Public surface:

* :func:`register_git` — wire the skill + factory onto an agent.
* :func:`make_git_handler` — build a termish ``CommandFunc`` directly
  from a kvgit substrate (used by tests and external callers).
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING, Any

from termish.context import CommandContext, CommandResult
from termish.errors import TerminalError

from .core import (
    AgentGitError,
    VirtualGit,
    is_binary,
)
from .refs import InvalidRef

if TYPE_CHECKING:
    from kvgit import Staged, VersionedKV
    from termish.errors import CommandFunc


# ---------------------------------------------------------------------------
# Public registration
# ---------------------------------------------------------------------------


def register_git(agent: Any) -> None:
    """Register the git skill + git terminal command on an agent.

    Mounts the git usage guide at ``/skills/git/SKILL.md`` for
    discovery via ``cat /skills/git/SKILL.md``, and registers the
    ``git`` terminal command so the agent can run git operations
    inside ``terminal_action`` pipelines.

    The terminal command is registered at ``visibility="low"`` —
    agents already know git from training, and the on-demand skill
    file is the in-depth reference; spending primer tokens on a brief
    description would be wasteful.

    Args:
        agent: An :class:`~agex.Agent` instance.
    """
    skill_bytes = resources.files("agex.skills").joinpath("git.md").read_bytes()
    agent.skill(skill_bytes)
    agent._terminal_command_factory(
        "git",
        _make_git_factory(),
        visibility="low",
        docstring=(
            "Git-style commit / branch / diff operations on the agent's VFS.  "
            "Run `git` with no args for usage; see /skills/git/SKILL.md for details."
        ),
    )


def _make_git_factory():
    """Per-action factory wired through the internal terminal API.

    Builds a fresh :class:`VirtualGit` per ``terminal_action`` because
    each invocation receives its own ``TerminalRuntime`` (fresh state
    + vfs).  ``VirtualGit`` itself is cheap to construct — it just
    holds references — so re-creation per pipeline is fine.
    """
    from agex.terminal import TerminalRuntime

    def factory(rt: "TerminalRuntime"):
        vkv = getattr(rt.state, "_versioned", None) if rt.state is not None else None
        if vkv is None:

            def termish_no_state(cmd_ctx):
                return CommandResult(
                    exit_code=1, stderr="git: requires versioned state"
                )

            return termish_no_state

        # Unwrap MountFS to expose the raw VFS, which has the
        # encode/decode helpers VirtualGit uses.  Mount is a transparent
        # composition layer — we want the underlying ``VirtualFS``.
        raw_vfs = rt.vfs
        if hasattr(rt.vfs, "_base"):
            raw_vfs = rt.vfs._base

        return make_git_handler(vkv, state=rt.state, vfs=raw_vfs)

    return factory


def make_git_handler(
    vkv: "VersionedKV",
    state: "Staged | None" = None,
    vfs: Any | None = None,
) -> "CommandFunc":
    """Build a termish ``CommandFunc`` that dispatches ``git`` subcommands.

    Args:
        vkv: The :class:`kvgit.VersionedKV` backing the agent.
        state: An optional :class:`kvgit.Staged` over ``vkv``.  When
            omitted, a fresh ``Staged(vkv)`` is constructed — convenient
            for ad-hoc usage from scripts but production callers should
            always pass the agent's actual ``Staged``.
        vfs: Optional :class:`monkeyfs.VirtualFS`.  When provided,
            user paths are translated to/from internal kvgit keys.

    Returns:
        A termish ``CommandFunc`` ready to be passed to ``execute()``
        as one of the ``commands`` entries.
    """
    if state is None:
        from kvgit import Staged

        state = Staged(vkv)
    vg = VirtualGit(vkv, state, vfs=vfs)
    return _build_dispatch(vg)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _build_dispatch(vg: VirtualGit) -> "CommandFunc":
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
        "rm": _git_rm,
    }

    def handler(ctx: CommandContext) -> CommandResult | None:
        args = ctx.args
        if not args:
            ctx.stdout.write(_usage())
            return None

        subcommand = args[0]
        subargs = args[1:]
        fn = dispatch.get(subcommand)
        if fn is None:
            raise TerminalError(f"git: '{subcommand}' is not a git command.")
        fn(subargs, ctx, vg)
        return None

    return handler


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
        "   add        Stage files for the next commit\n"
        "   rm         Remove files from the workspace\n"
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _git_log(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    oneline = "--oneline" in args
    max_count: int | None = None
    path_filter: str | None = None

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
            pass
        elif not arg.startswith("-") and arg != "--":
            path_filter = arg
        i += 1

    try:
        commits = vg.log(max_count=max_count, path=path_filter)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git log: {e}")

    branch = vg.current_branch
    head = vg.head()

    for c in commits:
        if oneline:
            head_marker = f" (HEAD -> {branch})" if c.hash == head else ""
            ctx.stdout.write(f"{c.short_hash}{head_marker} {c.message}\n")
        else:
            ctx.stdout.write(f"commit {c.hash}\n")
            if c.hash == head:
                ctx.stdout.write(f"  (HEAD -> {branch})\n")
            ctx.stdout.write(f"\n    {c.message}\n\n")


def _git_diff(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    refs: list[str] = []
    path_filter: str | None = None
    past_separator = False
    for arg in args:
        if arg == "--":
            past_separator = True
        elif past_separator:
            path_filter = arg
        elif not arg.startswith("-"):
            refs.append(arg)

    if len(refs) > 2:
        raise TerminalError("git diff: too many arguments")

    try:
        if len(refs) == 0:
            output = vg.diff(path=path_filter)
        elif len(refs) == 1:
            a = vg.resolve_ref(refs[0])
            output = vg.diff(a, None, path=path_filter)
        else:
            a = vg.resolve_ref(refs[0])
            b = vg.resolve_ref(refs[1])
            output = vg.diff(a, b, path=path_filter)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git diff: {e}")

    ctx.stdout.write(output)


def _git_status(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    s = vg.status()
    ctx.stdout.write(f"On branch {s.branch}\n")

    if s.staged:
        ctx.stdout.write("\nChanges to be committed:\n")
        for f in s.staged:
            ctx.stdout.write(f"  {f}\n")

    if s.unstaged:
        ctx.stdout.write("\nChanges not staged for commit:\n")
        ctx.stdout.write("  (use `git add <file>` to stage)\n")
        for f in s.unstaged:
            ctx.stdout.write(f"  {f}\n")

    if s.is_clean:
        ctx.stdout.write("nothing to commit, working tree clean\n")

    # Recent commits — best-effort; an unborn branch / corrupt store
    # should still let `git status` emit the branch header.
    try:
        recent = vg.log(max_count=3)
    except (AgentGitError, InvalidRef):
        recent = []
    if recent:
        ctx.stdout.write("\nRecent commits:\n")
        for c in recent:
            ctx.stdout.write(f"  {c.short_hash} {c.message}\n")


def _git_branch(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    if not args:
        current = vg.current_branch
        for b in vg.list_branches():
            marker = "* " if b == current else "  "
            ctx.stdout.write(f"{marker}{b}\n")
        return

    if args[0] in ("-d", "-D"):
        force = args[0] == "-D"
        if len(args) < 2:
            raise TerminalError("git branch: branch name required")
        name = args[1]
        try:
            vg.delete_branch(name, force=force)
        except (AgentGitError, InvalidRef) as e:
            raise TerminalError(f"git branch: {e}")
        ctx.stdout.write(f"Deleted branch {name}\n")
        return

    name = args[0]
    try:
        vg.create_branch(name)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git branch: {e}")
    ctx.stdout.write(f"Created branch {name}\n")


def _git_checkout(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    if not args:
        raise TerminalError("git checkout: branch name required")

    create = False
    force = False
    name: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-b":
            create = True
            i += 1
            if i >= len(args):
                raise TerminalError("git checkout: branch name required after -b")
            name = args[i]
        elif arg == "-f":
            force = True
        elif not arg.startswith("-") and name is None:
            name = arg
        i += 1

    if name is None:
        raise TerminalError("git checkout: branch name required")

    try:
        vg.checkout(name, create=create, force=force)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git checkout: {e}")

    if create:
        ctx.stdout.write(f"Switched to a new branch '{name}'\n")
    else:
        ctx.stdout.write(f"Switched to branch '{name}'\n")


def _git_commit(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    message: str | None = None
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

    try:
        c = vg.commit(message)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git commit: {e}")

    branch = c.virtual_branch or vg.current_branch
    ctx.stdout.write(f"[{branch} {c.short_hash}] {c.message}\n")


def _git_reset(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    hard = "--hard" in args
    refs = [a for a in args if not a.startswith("-")]

    if not hard:
        raise TerminalError("git reset: only --hard is supported")
    if not refs:
        raise TerminalError("git reset: need a ref (e.g. HEAD~1)")

    try:
        target = vg.resolve_ref(refs[0])
        vg.reset(target, hard=True)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git reset: {e}")

    ctx.stdout.write(f"Restored files to {target[:7]}\n")


def _git_show(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    if not args:
        raise TerminalError("git show: need a ref (e.g. HEAD:path/to/file)")

    ref_path = args[0]
    if ":" not in ref_path:
        raise TerminalError(
            "git show: use <ref>:<path> format (e.g. HEAD:helpers/utils.py)"
        )
    ref, path = ref_path.split(":", 1)

    try:
        commit_hash = vg.resolve_ref(ref or "HEAD")
        content = vg.show(commit_hash, path)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git show: {e}")
    except FileNotFoundError as e:
        raise TerminalError(f"git show: {e}")

    if is_binary(content):
        ctx.stdout.write(f"(binary file: {path}, {len(content)} bytes)\n")
    else:
        ctx.stdout.write(content.decode("utf-8", errors="replace"))


def _git_merge(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    if not args:
        raise TerminalError("git merge: branch name required")

    source = args[0]
    try:
        result = vg.merge(source)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git merge: {e}")

    if result is None:
        ctx.stdout.write("Already up to date.\n")
        return

    ctx.stdout.write(f"Merge made: {result.short_hash} {result.message}\n")


def _git_add(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    if not args:
        raise TerminalError("git add: nothing specified")
    try:
        vg.add(list(args))
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git add: {e}")


def _git_rm(args: list[str], ctx: CommandContext, vg: VirtualGit) -> None:
    if not args:
        raise TerminalError("git rm: nothing specified")

    recursive = "-r" in args
    paths = [a for a in args if not a.startswith("-")]
    if not paths:
        raise TerminalError("git rm: nothing specified")

    try:
        # Track what's about to vanish so we can echo per-file output.
        # ``vg.rm`` is silent by design (matches the existing CLI's
        # behaviour by only emitting the final ``rm '<path>'`` lines).
        emitted: list[str] = []
        for path in paths:
            vg.rm([path], recursive=recursive)
            emitted.append(path)
    except (AgentGitError, InvalidRef) as e:
        raise TerminalError(f"git rm: {e}")

    for name in sorted(emitted):
        ctx.stdout.write(f"rm '{name}'\n")
