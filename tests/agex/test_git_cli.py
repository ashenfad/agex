"""Tests for the git CLI adapter at ``agex/agent_git/cli.py``.

The bulk of agent-git semantic correctness is verified directly
against :class:`VirtualGit` in ``tests/agex/agent_git/``.  This file
focuses on the thin CLI surface:

* Per-subcommand argument parsing (flags, positionals, ``--`` separator).
* Output formatting — the exact strings the agent observes on stdout.
* Error translation to :class:`TerminalError` with informative prefixes.
* Pipeline composition with termish builtins (``grep`` / ``wc`` / ``head``).
* Full-stack integration through :class:`monkeyfs.VirtualFS` and
  termish's ``execute()`` — paths round-trip cleanly, no internal
  ``__vfs_`` keys leak into agent-visible output.
* Isolation contracts observable at the CLI level — branch / checkout /
  merge through ``git`` must NOT touch the kvgit physical branch or
  non-VFS keys (the original bug this refactor exists to fix).

When something fails, prefer fixing the test if it's checking
implementation detail (e.g., a closure variable that no longer exists)
and the code if it's checking observable behaviour (e.g., output
contents, error messages, isolation guarantees).
"""

from __future__ import annotations

import io

import pytest
from kvgit import Staged
from kvgit.store import Memory
from kvgit.versioned.kv import VersionedKV
from termish import MemoryFS, execute
from termish.context import CommandContext
from termish.errors import TerminalError

from agex.agent_git.cli import make_git_handler

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def vkv():
    return VersionedKV(Memory())


@pytest.fixture
def state(vkv):
    return Staged(vkv)


@pytest.fixture
def git(vkv, state):
    """Git command handler (no VFS — keys are user-visible verbatim)."""
    return make_git_handler(vkv, state=state)


@pytest.fixture
def vfs_setup(vkv, state):
    """Full stack: VersionedKV + Staged + VirtualFS + git handler.

    Returns a tuple ``(vkv, state, vfs, git_handler)``.  Use this when
    the test needs encoded paths or needs to exercise the
    ``__vfs_metadata__`` / CWD filtering.
    """
    from monkeyfs import VirtualFS

    vfs = VirtualFS(state)
    handler = make_git_handler(vkv, state=state, vfs=vfs)
    return vkv, state, vfs, handler


def run_git(handler, *args, stdin: str = "") -> str:
    """Invoke a git handler with positional args; return collected stdout.

    Equivalent to typing ``git <args...>`` at a prompt.  TerminalErrors
    propagate naturally so ``with pytest.raises`` works as expected.
    """
    stdout = io.StringIO()
    ctx = CommandContext(
        args=list(args),
        stdin=io.StringIO(stdin),
        stdout=stdout,
        fs=MemoryFS(),
    )
    handler(ctx)
    return stdout.getvalue()


def cli_commit(handler, state, files: dict, message: str) -> str:
    """Stage ``files`` directly and commit through the CLI.

    Mirrors the production flow where files arrive in Staged via
    other means (FILE actions, terminal redirection) and then
    ``git commit -m`` records them as a virtual commit.
    """
    for key, value in files.items():
        state[key] = value
    return run_git(handler, "commit", "-m", message)


# ===========================================================================
# Usage / dispatch
# ===========================================================================


class TestUsage:
    def test_no_args_prints_usage(self, git):
        out = run_git(git)
        assert out.startswith("usage: git")
        # Every subcommand must appear in the usage text so the agent
        # can read the help via `git` and learn what's available.
        for sub in (
            "log",
            "diff",
            "status",
            "branch",
            "checkout",
            "commit",
            "reset",
            "show",
            "merge",
            "add",
            "rm",
        ):
            assert sub in out

    def test_unknown_subcommand_raises(self, git):
        with pytest.raises(TerminalError, match="not a git command"):
            run_git(git, "stash")
        with pytest.raises(TerminalError, match="not a git command"):
            run_git(git, "rebase")


# ===========================================================================
# git status
# ===========================================================================


class TestStatus:
    def test_unborn_branch(self, git):
        out = run_git(git, "status")
        assert "On branch main" in out
        assert "nothing to commit" in out

    def test_after_first_commit(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        out = run_git(git, "status")
        assert "On branch main" in out
        assert "nothing to commit" in out
        assert "Recent commits" in out
        assert "init" in out

    def test_unstaged_modification(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        state["a"] = b"2"
        out = run_git(git, "status")
        assert "Changes not staged for commit" in out
        assert "(use `git add" in out
        assert " a" in out  # decoded path appears, indented

    def test_staged_modification_after_add(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        state["a"] = b"2"
        run_git(git, "add", "a")
        out = run_git(git, "status")
        assert "Changes to be committed" in out
        assert "Changes not staged" not in out

    def test_split_staged_unstaged(self, git, state):
        cli_commit(git, state, {"a": b"1", "b": b"1"}, "init")
        state["a"] = b"2"
        state["b"] = b"2"
        run_git(git, "add", "a")
        out = run_git(git, "status")
        assert "Changes to be committed" in out
        assert "Changes not staged for commit" in out

    def test_status_recent_commits_capped_at_three(self, git, state):
        for i in range(5):
            cli_commit(git, state, {"f": str(i).encode()}, f"commit {i}")
        out = run_git(git, "status")
        assert "commit 4" in out  # most recent
        assert "commit 2" in out
        assert "commit 1" not in out  # 4th-most-recent is dropped

    def test_status_branch_after_checkout(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "checkout", "-b", "feature")
        out = run_git(git, "status")
        assert "On branch feature" in out


# ===========================================================================
# git log
# ===========================================================================


class TestLog:
    def test_unborn_emits_nothing(self, git):
        # Real git errors here ("does not have any commits yet"); we
        # take a softer route and emit nothing — log on an unborn
        # branch is well-defined as "no history yet".
        assert run_git(git, "log") == ""

    def test_log_full_format(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "first")
        out = run_git(git, "log")
        assert "commit " in out
        assert "(HEAD -> main)" in out
        assert "first" in out

    def test_log_oneline(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "first")
        cli_commit(git, state, {"a": b"2"}, "second")
        out = run_git(git, "log", "--oneline")
        lines = [line for line in out.strip().split("\n") if line]
        assert len(lines) == 2
        # Most recent first
        assert "second" in lines[0]
        assert "first" in lines[1]
        # 7-char short hash on each line
        for line in lines:
            assert len(line.split()[0]) == 7

    def test_log_oneline_head_marker_only_on_tip(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "first")
        cli_commit(git, state, {"a": b"2"}, "second")
        out = run_git(git, "log", "--oneline")
        lines = [line for line in out.strip().split("\n") if line]
        assert "(HEAD -> main)" in lines[0]
        assert "(HEAD -> main)" not in lines[1]

    def test_log_max_count_long_flag(self, git, state):
        for i in range(5):
            cli_commit(git, state, {"f": str(i).encode()}, f"commit {i}")
        out = run_git(git, "log", "--oneline", "--max-count", "2")
        assert len([line for line in out.strip().split("\n") if line]) == 2

    def test_log_max_count_short_separated(self, git, state):
        for i in range(5):
            cli_commit(git, state, {"f": str(i).encode()}, f"commit {i}")
        out = run_git(git, "log", "--oneline", "-n", "3")
        assert len([line for line in out.strip().split("\n") if line]) == 3

    def test_log_max_count_short_attached(self, git, state):
        for i in range(5):
            cli_commit(git, state, {"f": str(i).encode()}, f"commit {i}")
        out = run_git(git, "log", "--oneline", "-n2")
        assert len([line for line in out.strip().split("\n") if line]) == 2

    def test_log_invalid_count_raises(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="invalid count"):
            run_git(git, "log", "-n", "abc")

    def test_log_path_filter(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "touched a")
        cli_commit(git, state, {"b": b"2"}, "touched b")
        out = run_git(git, "log", "--oneline", "a")
        assert "touched a" in out
        assert "touched b" not in out


# ===========================================================================
# git diff
# ===========================================================================


class TestDiff:
    def test_diff_no_args_clean(self, git, state):
        cli_commit(git, state, {"a": b"1\n"}, "init")
        assert run_git(git, "diff").strip() == ""

    def test_diff_no_args_unborn_silent(self, git):
        assert run_git(git, "diff") == ""

    def test_diff_no_args_pending(self, git, state):
        cli_commit(git, state, {"a": b"old\n"}, "init")
        state["a"] = b"new\n"
        out = run_git(git, "diff")
        assert "-old" in out
        assert "+new" in out

    def test_diff_one_ref_vs_working(self, git, state):
        cli_commit(git, state, {"a": b"v1\n"}, "v1")
        cli_commit(git, state, {"a": b"v2\n"}, "v2")
        # `git diff HEAD~1` = HEAD~1 vs working tree
        out = run_git(git, "diff", "HEAD~1")
        assert "-v1" in out
        assert "+v2" in out

    def test_diff_two_refs(self, git, state):
        cli_commit(git, state, {"a": b"v1\n"}, "v1")
        cli_commit(git, state, {"a": b"v2\n"}, "v2")
        out = run_git(git, "diff", "HEAD~1", "HEAD")
        assert "-v1" in out
        assert "+v2" in out

    def test_diff_path_filter_with_separator(self, git, state):
        cli_commit(git, state, {"a": b"a1\n", "b": b"b1\n"}, "v1")
        cli_commit(git, state, {"a": b"a2\n", "b": b"b2\n"}, "v2")
        out = run_git(git, "diff", "HEAD~1", "HEAD", "--", "a")
        assert "a/a" in out
        assert "b/b" not in out

    def test_diff_too_many_args(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="too many"):
            run_git(git, "diff", "HEAD", "HEAD", "HEAD")

    def test_diff_invalid_ref(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="git diff"):
            run_git(git, "diff", "nonexistent")

    def test_diff_added_file_appears(self, git, state):
        cli_commit(git, state, {"a": b"1\n"}, "init")
        cli_commit(git, state, {"a": b"1\n", "b": b"new\n"}, "add b")
        out = run_git(git, "diff", "HEAD~1", "HEAD")
        assert "b/b" in out
        assert "+new" in out

    def test_diff_removed_file_shows_minus(self, git, state):
        cli_commit(git, state, {"a": b"1\n", "b": b"2\n"}, "init")
        del state["b"]
        run_git(git, "commit", "-m", "remove b")
        out = run_git(git, "diff", "HEAD~1", "HEAD")
        assert "a/b" in out
        assert "-2" in out

    def test_diff_binary_files_summary(self, git, state):
        cli_commit(git, state, {"f": b"\x00\x01"}, "v1")
        cli_commit(git, state, {"f": b"\x00\x02"}, "v2")
        out = run_git(git, "diff", "HEAD~1", "HEAD")
        assert "Binary files" in out


# ===========================================================================
# git show
# ===========================================================================


class TestShow:
    def test_show_at_head(self, git, state):
        cli_commit(git, state, {"a": b"hello"}, "init")
        assert run_git(git, "show", "HEAD:a") == "hello"

    def test_show_at_older_commit(self, git, state):
        cli_commit(git, state, {"f": b"old"}, "v1")
        cli_commit(git, state, {"f": b"new"}, "v2")
        assert run_git(git, "show", "HEAD~1:f") == "old"

    def test_show_by_short_hash(self, git, state):
        cli_commit(git, state, {"f": b"v1"}, "v1")
        c2 = cli_commit(git, state, {"f": b"v2"}, "v2")
        # Extract the short hash from the commit-line output
        # ([main 1234567] v2)
        short = c2.split()[1].rstrip("]")
        assert run_git(git, "show", f"{short}:f") == "v2"

    def test_show_missing_path(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="not found"):
            run_git(git, "show", "HEAD:nope")

    def test_show_invalid_ref_format(self, git):
        with pytest.raises(TerminalError, match="<ref>:<path>"):
            run_git(git, "show", "HEAD")

    def test_show_no_args(self, git):
        with pytest.raises(TerminalError, match="ref"):
            run_git(git, "show")

    def test_show_binary_summary(self, git, state):
        cli_commit(git, state, {"f": b"\x00\x01\x02hello"}, "v1")
        out = run_git(git, "show", "HEAD:f")
        assert "binary file" in out


# ===========================================================================
# git branch
# ===========================================================================


class TestBranch:
    def test_list_lists_only_main_initially(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        out = run_git(git, "branch")
        assert "* main" in out

    def test_list_after_create(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "branch", "feature")
        out = run_git(git, "branch")
        assert "* main" in out
        assert "  feature" in out

    def test_list_marks_only_current(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "branch", "feature")
        out = run_git(git, "branch")
        # Exactly one line begins with the active marker.
        marker_count = sum(1 for line in out.split("\n") if line.startswith("* "))
        assert marker_count == 1

    def test_create_announces(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        out = run_git(git, "branch", "feat")
        assert "Created branch feat" in out

    def test_create_unborn_fails(self, git):
        with pytest.raises(TerminalError, match="git branch"):
            run_git(git, "branch", "feat")

    def test_create_duplicate_fails(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "branch", "dup")
        with pytest.raises(TerminalError, match="already exists"):
            run_git(git, "branch", "dup")

    def test_delete_safe(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "branch", "tmp")
        out = run_git(git, "branch", "-d", "tmp")
        assert "Deleted branch tmp" in out

    def test_delete_missing_fails(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="not found"):
            run_git(git, "branch", "-d", "ghost")

    def test_delete_current_fails(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="git branch"):
            run_git(git, "branch", "-d", "main")

    def test_delete_unmerged_fails(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "checkout", "-b", "feat")
        cli_commit(git, state, {"b": b"feat"}, "feat work")
        run_git(git, "checkout", "main")
        with pytest.raises(TerminalError, match="not fully merged"):
            run_git(git, "branch", "-d", "feat")

    def test_force_delete_unmerged(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "checkout", "-b", "feat")
        cli_commit(git, state, {"b": b"feat"}, "feat work")
        run_git(git, "checkout", "main")
        out = run_git(git, "branch", "-D", "feat")
        assert "Deleted" in out

    def test_delete_no_branch_name_fails(self, git):
        with pytest.raises(TerminalError, match="branch name required"):
            run_git(git, "branch", "-d")


# ===========================================================================
# git checkout
# ===========================================================================


class TestCheckout:
    def test_no_args(self, git):
        with pytest.raises(TerminalError, match="branch name required"):
            run_git(git, "checkout")

    def test_switch_existing(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "branch", "dev")
        out = run_git(git, "checkout", "dev")
        assert "Switched to branch 'dev'" in out

    def test_switch_missing_fails(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="git checkout"):
            run_git(git, "checkout", "ghost")

    def test_dash_b_creates_and_switches(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        out = run_git(git, "checkout", "-b", "feature")
        assert "Switched to a new branch 'feature'" in out

    def test_dash_b_requires_name(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="-b"):
            run_git(git, "checkout", "-b")

    def test_pending_changes_refused(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "branch", "feat")
        state["a"] = b"dirty"
        with pytest.raises(TerminalError, match="local changes"):
            run_git(git, "checkout", "feat")

    def test_force_discards_pending(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "branch", "feat")
        state["a"] = b"dirty"
        out = run_git(git, "checkout", "-f", "feat")
        assert "Switched" in out
        assert state.get("a") == b"1"  # branch tip's value

    def test_checkout_applies_target_view(self, git, state):
        cli_commit(git, state, {"a": b"main"}, "init")
        run_git(git, "checkout", "-b", "feat")
        cli_commit(git, state, {"b": b"feat"}, "feat add")
        run_git(git, "checkout", "main")
        # b should NOT exist on main
        assert "b" not in state


# ===========================================================================
# git commit
# ===========================================================================


class TestCommit:
    def test_commit_announces(self, git, state):
        state["a"] = b"1"
        out = run_git(git, "commit", "-m", "init")
        assert out.startswith("[main ")
        assert "init" in out

    def test_commit_message_attached_form(self, git, state):
        state["a"] = b"1"
        # `-mfoo` should also work (attached value)
        out = run_git(git, "commit", "-mfoo bar")
        assert "foo bar" in out

    def test_commit_no_message_fails(self, git, state):
        state["a"] = b"1"
        with pytest.raises(TerminalError, match="message"):
            run_git(git, "commit")

    def test_commit_dash_m_without_value_fails(self, git, state):
        state["a"] = b"1"
        with pytest.raises(TerminalError, match="-m"):
            run_git(git, "commit", "-m")

    def test_commit_nothing_pending_fails(self, git, state):
        state["a"] = b"1"
        run_git(git, "commit", "-m", "init")
        with pytest.raises(TerminalError, match="nothing to commit"):
            run_git(git, "commit", "-m", "empty")

    def test_commit_advances_history(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "first")
        cli_commit(git, state, {"a": b"2"}, "second")
        out = run_git(git, "log", "--oneline")
        lines = [line for line in out.strip().split("\n") if line]
        assert len(lines) == 2

    def test_commit_short_hash_in_output(self, git, state):
        state["a"] = b"1"
        out = run_git(git, "commit", "-m", "init")
        # [main <7-char> init]
        token = out.split()[1].rstrip("]")
        assert len(token) == 7


# ===========================================================================
# git add / git rm
# ===========================================================================


class TestAdd:
    def test_no_args_fails(self, git):
        with pytest.raises(TerminalError, match="nothing"):
            run_git(git, "add")

    def test_add_unknown_path_fails(self, git):
        with pytest.raises(TerminalError, match="did not match"):
            run_git(git, "add", "ghost.py")

    def test_add_specific_path(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        state["a"] = b"2"
        run_git(git, "add", "a")
        # Status should now show the file as staged
        out = run_git(git, "status")
        assert "Changes to be committed" in out

    def test_add_dot_stages_all(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        state["a"] = b"2"
        state["b"] = b"new"
        run_git(git, "add", ".")
        out = run_git(git, "status")
        assert "Changes to be committed" in out
        assert "Changes not staged" not in out

    def test_add_dash_a_alias(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        state["a"] = b"2"
        run_git(git, "add", "-A")
        out = run_git(git, "status")
        assert "Changes to be committed" in out

    def test_selective_commit_only_flushes_index(self, git, state):
        state["a"] = b"a"
        state["b"] = b"b"
        run_git(git, "add", "a")
        run_git(git, "commit", "-m", "just a")
        # b is still pending (modified vs branch tip == b)
        out = run_git(git, "status")
        assert " b" in out


class TestRm:
    def test_no_args_fails(self, git):
        with pytest.raises(TerminalError, match="nothing"):
            run_git(git, "rm")

    def test_rm_announces_per_path(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        out = run_git(git, "rm", "a")
        assert "rm 'a'" in out

    def test_rm_removes_from_state(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "rm", "a")
        assert "a" not in state

    def test_rm_missing_fails(self, git):
        with pytest.raises(TerminalError, match="did not match"):
            run_git(git, "rm", "ghost")

    def test_rm_then_commit_persists_deletion(self, git, state):
        cli_commit(git, state, {"a": b"1", "b": b"2"}, "init")
        run_git(git, "rm", "a")
        run_git(git, "commit", "-m", "remove a")
        out = run_git(git, "log", "--oneline")
        assert "remove a" in out


# ===========================================================================
# git reset
# ===========================================================================


class TestReset:
    def test_only_hard(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="--hard"):
            run_git(git, "reset", "HEAD")

    def test_no_ref_fails(self, git):
        with pytest.raises(TerminalError, match="ref"):
            run_git(git, "reset", "--hard")

    def test_reset_hard_restores_files(self, git, state):
        cli_commit(git, state, {"f": b"v1"}, "v1")
        cli_commit(git, state, {"f": b"v2"}, "v2")
        out = run_git(git, "reset", "--hard", "HEAD~1")
        assert "Restored" in out
        assert state.get("f") == b"v1"

    def test_reset_rewinds_branch_ref(self, git, state):
        cli_commit(git, state, {"f": b"v1"}, "v1")
        cli_commit(git, state, {"f": b"v2"}, "v2")
        run_git(git, "reset", "--hard", "HEAD~1")
        out = run_git(git, "log", "--oneline")
        assert "v1" in out
        assert "v2" not in out  # rewound past it

    def test_reset_does_not_lose_kvgit_history(self, git, state, vkv):
        cli_commit(git, state, {"f": b"v1"}, "v1")
        cli_commit(git, state, {"f": b"v2"}, "v2")
        history_before = list(vkv.history())
        run_git(git, "reset", "--hard", "HEAD~1")
        history_after = list(vkv.history())
        # No commits dropped from kvgit; reset is virtual.
        assert set(history_before).issubset(set(history_after))


# ===========================================================================
# git merge
# ===========================================================================


class TestMerge:
    def test_no_args_fails(self, git):
        with pytest.raises(TerminalError, match="branch name required"):
            run_git(git, "merge")

    def test_merge_missing_branch(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="not found"):
            run_git(git, "merge", "ghost")

    def test_merge_self_fails(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="itself"):
            run_git(git, "merge", "main")

    def test_merge_already_up_to_date(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "branch", "same")
        out = run_git(git, "merge", "same")
        assert "Already up to date" in out

    def test_fast_forward(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "checkout", "-b", "feat")
        cli_commit(git, state, {"b": b"feat"}, "feat work")
        run_git(git, "checkout", "main")
        out = run_git(git, "merge", "feat")
        assert "Merge made" in out  # CLI announces

    def test_true_merge(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "checkout", "-b", "feat")
        cli_commit(git, state, {"b": b"feat"}, "feat work")
        run_git(git, "checkout", "main")
        cli_commit(git, state, {"a": b"main_v2"}, "main work")
        out = run_git(git, "merge", "feat")
        assert "Merge made" in out
        assert state.get("a") == b"main_v2"  # main's local change preserved
        assert state.get("b") == b"feat"

    def test_merge_pending_changes_refused(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        run_git(git, "checkout", "-b", "feat")
        cli_commit(git, state, {"b": b"feat"}, "feat work")
        run_git(git, "checkout", "main")
        state["a"] = b"dirty"
        with pytest.raises(TerminalError, match="local changes"):
            run_git(git, "merge", "feat")


# ===========================================================================
# Pipeline composition with termish builtins
# ===========================================================================


class TestPipelineComposition:
    """``git`` must compose with termish's pipeline operators just like
    a real shell.  Each stage runs in sequence, stdout streaming to the
    next stage's stdin.
    """

    def test_log_piped_to_grep(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "add feature X")
        cli_commit(git, state, {"b": b"2"}, "fix bug Y")
        cli_commit(git, state, {"c": b"3"}, "add feature Z")

        fs = MemoryFS()
        out = execute("git log --oneline | grep add", fs, commands={"git": git})
        lines = [line for line in out.strip().split("\n") if line]
        assert len(lines) == 2
        assert all("add" in line for line in lines)

    def test_log_piped_to_wc(self, git, state):
        for i in range(4):
            cli_commit(git, state, {"f": str(i).encode()}, f"commit {i}")
        fs = MemoryFS()
        out = execute("git log --oneline | wc -l", fs, commands={"git": git})
        assert int(out.strip()) == 4

    def test_log_piped_to_head(self, git, state):
        for i in range(5):
            cli_commit(git, state, {"f": str(i).encode()}, f"c{i}")
        fs = MemoryFS()
        out = execute("git log --oneline | head -2", fs, commands={"git": git})
        assert len([line for line in out.strip().split("\n") if line]) == 2

    def test_status_piped_to_grep_branch(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "init")
        fs = MemoryFS()
        out = execute("git status | grep 'On branch'", fs, commands={"git": git})
        assert "On branch main" in out


# ===========================================================================
# Full-stack integration via VirtualFS
# ===========================================================================


class TestVFSIntegration:
    """Termish writes (e.g. ``echo > foo``) → VirtualFS path encoding →
    Staged → kvgit, then ``git`` operations see clean user paths
    everywhere — no internal ``__vfs_`` keys leak.
    """

    def test_write_then_commit_through_vfs(self, vfs_setup):
        vkv, state, vfs, git = vfs_setup
        commands = {"git": git}
        execute("echo 'print(42)' > hello.py", vfs, commands=commands)
        out = execute("git commit -m 'initial hello'", vfs, commands=commands)
        assert "initial hello" in out
        assert (
            execute("git log --oneline", vfs, commands=commands).count("initial hello")
            == 1
        )

    def test_diff_shows_clean_paths(self, vfs_setup):
        vkv, state, vfs, git = vfs_setup
        commands = {"git": git}
        execute("echo 'v1' > app.py", vfs, commands=commands)
        execute("git commit -m 'v1'", vfs, commands=commands)
        execute("echo 'v2' > app.py", vfs, commands=commands)
        execute("git commit -m 'v2'", vfs, commands=commands)

        out = execute("git diff HEAD~1", vfs, commands=commands)
        assert "a/app.py" in out
        assert "b/app.py" in out
        # Crucial: no internal monkeyfs key prefix leaks to the agent.
        assert "__vfs_" not in out

    def test_show_decodes_paths(self, vfs_setup):
        vkv, state, vfs, git = vfs_setup
        commands = {"git": git}
        execute("echo 'old' > data.py", vfs, commands=commands)
        execute("git commit -m 'old'", vfs, commands=commands)
        execute("echo 'new' > data.py", vfs, commands=commands)
        execute("git commit -m 'new'", vfs, commands=commands)
        assert "old" in execute("git show HEAD~1:data.py", vfs, commands=commands)

    def test_branch_workflow_through_vfs(self, vfs_setup):
        vkv, state, vfs, git = vfs_setup
        commands = {"git": git}
        execute("echo 'base' > shared.py", vfs, commands=commands)
        execute("git commit -m 'base'", vfs, commands=commands)
        execute("git checkout -b feat", vfs, commands=commands)
        execute("echo 'feat' > new.py", vfs, commands=commands)
        execute("git commit -m 'feat work'", vfs, commands=commands)
        execute("git checkout main", vfs, commands=commands)
        # Feature's file should NOT exist on main
        assert not vfs.exists("new.py")
        # Merge brings it back
        execute("git merge feat", vfs, commands=commands)
        assert vfs.exists("new.py")

    def test_reset_restores_through_vfs(self, vfs_setup):
        vkv, state, vfs, git = vfs_setup
        commands = {"git": git}
        execute("echo 'v1' > code.py", vfs, commands=commands)
        execute("git commit -m 'v1'", vfs, commands=commands)
        execute("echo 'v2' > code.py", vfs, commands=commands)
        execute("git commit -m 'v2'", vfs, commands=commands)
        execute("git reset --hard HEAD~1", vfs, commands=commands)
        # ``echo`` appends a newline, hence the trailing ``\n``.
        assert vfs.read("code.py") == b"v1\n"

    def test_status_excludes_vfs_metadata(self, vfs_setup):
        # ``__vfs_metadata__`` gets touched on every VFS write but it's
        # a substrate concern — must NEVER appear in `git status`.
        vkv, state, vfs, git = vfs_setup
        commands = {"git": git}
        execute("echo 'x' > a.py", vfs, commands=commands)
        out = execute("git status", vfs, commands=commands)
        assert "__vfs" not in out


# ===========================================================================
# Isolation contracts at the CLI level
# ===========================================================================


class TestIsolationContracts:
    """The original bug: ``git`` operations must not move the kvgit
    physical branch or destroy non-VFS substrate state.  These tests
    drive the contracts through the CLI surface so a regression at any
    layer (CLI parsing, VirtualGit, refs) is caught.
    """

    def _stub_vfs(self):
        """Minimal VFS-shaped object used to exercise visibility filtering
        without pulling in monkeyfs's path encoder.  Treats
        ``__vfs_<name>`` keys as files and everything else as substrate.
        """

        class _StubVFS:
            PREFIX = "__vfs_"
            METADATA_KEY = "__vfs_metadata__"
            CWD_KEY = "__vfs_cwd__"

            def _is_vfs_key(self, key):
                return key.startswith(self.PREFIX)

            def _encode_path(self, path):
                return self.PREFIX + path

            def _decode_path(self, key):
                return key[len(self.PREFIX) :]

        return _StubVFS()

    def test_create_branch_does_not_create_kvgit_branch(self, vkv, state):
        git = make_git_handler(vkv, state=state, vfs=self._stub_vfs())
        state["__vfs_a"] = b"1"
        run_git(git, "commit", "-m", "init")
        kvgit_before = set(vkv.list_branches())
        run_git(git, "branch", "experiment")
        assert set(vkv.list_branches()) == kvgit_before

    def test_checkout_does_not_switch_kvgit_branch(self, vkv, state):
        git = make_git_handler(vkv, state=state, vfs=self._stub_vfs())
        state["__vfs_a"] = b"1"
        run_git(git, "commit", "-m", "init")
        run_git(git, "branch", "feat")
        kvgit_branch_before = vkv.current_branch
        run_git(git, "checkout", "feat")
        assert vkv.current_branch == kvgit_branch_before

    def test_checkout_preserves_event_log_keys(self, vkv, state):
        git = make_git_handler(vkv, state=state, vfs=self._stub_vfs())
        state["__vfs_a"] = b"1"
        run_git(git, "commit", "-m", "init")
        run_git(git, "branch", "feat")

        # Seed substrate-style keys (mimics framework state)
        state["__event_log__/0"] = b"event-A"
        state["repl/x"] = b"repl-value"

        run_git(git, "checkout", "feat")
        assert state.get("__event_log__/0") == b"event-A"
        assert state.get("repl/x") == b"repl-value"

    def test_merge_preserves_event_log_keys(self, vkv, state):
        git = make_git_handler(vkv, state=state, vfs=self._stub_vfs())
        state["__vfs_a"] = b"1"
        run_git(git, "commit", "-m", "init")
        run_git(git, "checkout", "-b", "feat")
        state["__vfs_b"] = b"feat"
        run_git(git, "commit", "-m", "feat work")
        run_git(git, "checkout", "main")

        state["__event_log__/0"] = b"survives merge"
        state["repl/x"] = b"also survives"

        run_git(git, "merge", "feat")
        assert state.get("__event_log__/0") == b"survives merge"
        assert state.get("repl/x") == b"also survives"

    def test_reset_preserves_event_log_keys(self, vkv, state):
        git = make_git_handler(vkv, state=state, vfs=self._stub_vfs())
        state["__vfs_a"] = b"v1"
        run_git(git, "commit", "-m", "v1")
        state["__vfs_a"] = b"v2"
        run_git(git, "commit", "-m", "v2")

        state["__event_log__/0"] = b"important"
        run_git(git, "reset", "--hard", "HEAD~1")
        assert state.get("__event_log__/0") == b"important"

    def test_metadata_key_invisible_to_agent_status(self, vkv, state):
        git = make_git_handler(vkv, state=state, vfs=self._stub_vfs())
        state["__vfs_a"] = b"1"
        run_git(git, "commit", "-m", "init")
        # `git add` writes metadata; status must not mention the
        # internal __agex_git__ key as a "file".
        run_git(git, "add", "a")
        out = run_git(git, "status")
        assert "__agex_git__" not in out
        assert "__vfs_metadata__" not in out


# ===========================================================================
# CLI / VirtualGit interaction subtleties
# ===========================================================================


class TestSubtleBehaviours:
    def test_index_persists_across_handler_instances(self, vkv, state):
        # Regression: the old ``_tracked`` closure didn't persist across
        # terminal_action invocations, so ``git add`` followed by
        # ``git commit`` in *separate* commands silently lost the
        # index.  Metadata-backed index must survive new handlers.
        h1 = make_git_handler(vkv, state=state)
        cli_commit(h1, state, {"a": b"1", "b": b"1"}, "init")
        state["a"] = b"2"
        run_git(h1, "add", "a")

        # New handler, fresh closures — must still see the index.
        h2 = make_git_handler(vkv, state=state)
        out = run_git(h2, "status")
        assert "Changes to be committed" in out

    def test_empty_state_handler_works(self, vkv):
        # ``make_git_handler`` accepts a bare vkv and synthesises a
        # Staged.  Used by ad-hoc external callers and the CLI tests.
        handler = make_git_handler(vkv)
        out = run_git(handler, "status")
        assert "On branch main" in out

    def test_diff_from_unborn_branch(self, git):
        # ``git diff`` on a fresh repo is silent (no commits to diff).
        assert run_git(git, "diff") == ""

    def test_commit_message_with_special_chars(self, git, state):
        state["a"] = b"1"
        out = run_git(git, "commit", "-m", "msg with [brackets] and 'quotes'")
        assert "[brackets]" in out
        assert "'quotes'" in out

    def test_subcommand_dispatch_is_case_sensitive(self, git):
        # `git Log` is not `git log`.  Real git treats subcommand case
        # as significant; we should too, for predictable failures.
        with pytest.raises(TerminalError, match="not a git command"):
            run_git(git, "Log")

    def test_log_oneline_short_hash_length(self, git, state):
        cli_commit(git, state, {"a": b"1"}, "first")
        out = run_git(git, "log", "--oneline")
        first_token = out.split()[0]
        assert len(first_token) == 7  # short hash convention
