"""Tests for the git CLI handler (agex/git_cli.py).

Each test creates a VersionedKV with Memory storage and exercises
git subcommands via the handler directly using CommandContext.
"""

import io

import pytest
from kvgit import Staged
from kvgit.store import Memory
from kvgit.versioned.kv import VersionedKV
from termish import MemoryFS, execute
from termish.context import CommandContext
from termish.errors import TerminalError

from agex.git_cli import make_git_handler


@pytest.fixture
def vkv():
    """Fresh VersionedKV on an in-memory store."""
    return VersionedKV(Memory())


@pytest.fixture
def state(vkv):
    """Staged wrapper around the VersionedKV."""
    return Staged(vkv)


@pytest.fixture
def git(vkv, state):
    """Git command handler bound to the VersionedKV with Staged."""
    return make_git_handler(vkv, state=state)


def run_git(git_handler, *args):
    """Helper: invoke the git handler with args and return stdout."""
    stdout = io.StringIO()
    ctx = CommandContext(
        args=list(args),
        stdin=io.StringIO(),
        stdout=stdout,
        fs=MemoryFS(),
    )
    git_handler(ctx)
    return stdout.getvalue()


def commit_files(state, files, message=None):
    """Helper: stage files and commit through Staged (matches real usage)."""
    for key, value in files.items():
        state[key] = value
    info = {"message": message} if message else None
    state.commit(info=info)


# =============================================================================
# git commit + log
# =============================================================================


class TestCommitAndLog:
    def test_commit_with_message(self, vkv, state, git):
        commit_files(state, {"hello.py": b"print('hello')"}, "initial")
        state["hello.py"] = b"print('hello v2')"
        output = run_git(git, "commit", "-m", "second commit")
        assert "second commit" in output

    def test_log_shows_commits(self, vkv, state, git):
        commit_files(state, {"a.py": b"1"}, "first")
        commit_files(state, {"a.py": b"2"}, "second")
        output = run_git(git, "log", "--oneline")
        assert "second" in output
        assert "first" in output
        # Most recent first
        assert output.index("second") < output.index("first")

    def test_log_max_count(self, vkv, state, git):
        for i in range(5):
            commit_files(state, {f"f{i}": b"x"}, f"commit {i}")
        output = run_git(git, "log", "--oneline", "-n", "2")
        lines = [line for line in output.strip().split("\n") if line]
        assert len(lines) == 2

    def test_log_head_marker(self, vkv, state, git):
        commit_files(state, {"a": b"1"}, "one")
        output = run_git(git, "log", "--oneline")
        assert "(HEAD -> main)" in output

    def test_commit_requires_message(self, git):
        with pytest.raises(TerminalError, match="-m"):
            run_git(git, "commit")

    def test_log_path_filter(self, vkv, state, git):
        commit_files(state, {"a.py": b"1"}, "touched a")
        commit_files(state, {"b.py": b"2"}, "touched b")
        output = run_git(git, "log", "--oneline", "a.py")
        assert "touched a" in output
        assert "touched b" not in output


# =============================================================================
# git diff
# =============================================================================


class TestDiff:
    def test_diff_shows_changes(self, vkv, state, git):
        commit_files(state, {"hello.py": b"print('hello')\n"}, "v1")
        commit_files(state, {"hello.py": b"print('world')\n"}, "v2")
        output = run_git(git, "diff", "HEAD~1", "HEAD")
        assert "hello" in output
        assert "world" in output
        assert "---" in output  # unified diff markers
        assert "+++" in output

    def test_diff_no_args_diffs_head_vs_parent(self, vkv, state, git):
        commit_files(state, {"f.py": b"old\n"}, "v1")
        commit_files(state, {"f.py": b"new\n"}, "v2")
        output = run_git(git, "diff")
        assert "old" in output
        assert "new" in output

    def test_diff_added_file(self, vkv, state, git):
        commit_files(state, {"a.py": b"existing\n"}, "v1")
        commit_files(state, {"a.py": b"existing\n", "b.py": b"new file\n"}, "v2")
        output = run_git(git, "diff", "HEAD~1")
        assert "b.py" in output
        assert "new file" in output

    def test_diff_path_filter(self, vkv, state, git):
        commit_files(state, {"a.py": b"old_a\n", "b.py": b"old_b\n"}, "v1")
        commit_files(state, {"a.py": b"new_a\n", "b.py": b"new_b\n"}, "v2")
        output = run_git(git, "diff", "HEAD~1", "HEAD", "--", "a.py")
        assert "a.py" in output
        assert "b.py" not in output


# =============================================================================
# git status
# =============================================================================


class TestStatus:
    def test_status_shows_branch(self, vkv, git):
        output = run_git(git, "status")
        assert "On branch main" in output

    def test_status_shows_recent_commits(self, vkv, state, git):
        commit_files(state, {"a": b"1"}, "my commit")
        output = run_git(git, "status")
        assert "my commit" in output


# =============================================================================
# git branch
# =============================================================================


class TestBranch:
    def test_list_branches(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        vkv.create_branch("experiment")
        output = run_git(git, "branch")
        assert "* main" in output
        assert "experiment" in output

    def test_create_branch(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        output = run_git(git, "branch", "feature")
        assert "Created branch feature" in output
        assert "feature" in vkv.list_branches()

    def test_delete_branch(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        vkv.create_branch("temp")
        output = run_git(git, "branch", "-d", "temp")
        assert "Deleted branch temp" in output
        assert "temp" not in vkv.list_branches()

    def test_cannot_delete_current_branch(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        with pytest.raises(TerminalError, match="Cannot delete"):
            run_git(git, "branch", "-d", "main")

    def test_create_duplicate_branch_fails(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        vkv.create_branch("dup")
        with pytest.raises(TerminalError, match="already exists"):
            run_git(git, "branch", "dup")


# =============================================================================
# git checkout
# =============================================================================


class TestCheckout:
    def test_checkout_existing_branch(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        vkv.create_branch("dev")
        output = run_git(git, "checkout", "dev")
        assert "Switched to branch 'dev'" in output
        assert vkv.current_branch == "dev"

    def test_checkout_b_creates_and_switches(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        output = run_git(git, "checkout", "-b", "feature")
        assert "Switched to a new branch 'feature'" in output
        assert vkv.current_branch == "feature"

    def test_checkout_nonexistent_fails(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        with pytest.raises(TerminalError, match="does not exist"):
            run_git(git, "checkout", "nope")


# =============================================================================
# git reset
# =============================================================================


class TestReset:
    def test_reset_hard_restores_files_without_moving_head(self, vkv, state, git):
        """Reset restores files from a tagged commit without rewinding kvgit HEAD."""
        state["a"] = b"version1"
        state.commit(info={"message": "v1"})
        v1_hash = vkv.current_commit

        state["a"] = b"version2"
        state.commit(info={"message": "v2"})

        assert state.get("a") == b"version2"

        output = run_git(git, "reset", "--hard", "HEAD~1")
        assert "Restored" in output

        # Files restored to v1 content
        assert state.get("a") == b"version1"
        # But kvgit HEAD was NOT moved back — it's still at or past v2
        assert vkv.current_commit != v1_hash

    def test_reset_requires_hard(self, git):
        with pytest.raises(TerminalError, match="only --hard"):
            run_git(git, "reset", "HEAD~1")

    def test_reset_requires_ref(self, git):
        with pytest.raises(TerminalError, match="need a ref"):
            run_git(git, "reset", "--hard")


# =============================================================================
# git show
# =============================================================================


class TestShow:
    def test_show_file_at_head(self, vkv, state, git):
        commit_files(state, {"hello.py": b"print('hello')\n"}, "init")
        output = run_git(git, "show", "HEAD:hello.py")
        assert "print('hello')" in output

    def test_show_file_at_older_commit(self, vkv, state, git):
        commit_files(state, {"f.py": b"old content\n"}, "v1")
        commit_files(state, {"f.py": b"new content\n"}, "v2")
        output = run_git(git, "show", "HEAD~1:f.py")
        assert "old content" in output
        assert "new content" not in output

    def test_show_missing_file(self, vkv, state, git):
        commit_files(state, {"a": b"1"}, "init")
        with pytest.raises(TerminalError, match="not found"):
            run_git(git, "show", "HEAD:nope.py")

    def test_show_requires_colon_path(self, git):
        with pytest.raises(TerminalError, match="<ref>:<path>"):
            run_git(git, "show", "HEAD")


# =============================================================================
# git merge
# =============================================================================


class TestMerge:
    def test_merge_branch(self, vkv, state, git):
        commit_files(state, {"shared.py": b"base\n"}, "base")
        vkv.create_branch("feature")
        vkv.switch_branch("feature")
        commit_files(
            state, {"shared.py": b"base\n", "new.py": b"feature code\n"}, "feature work"
        )
        vkv.switch_branch("main")

        output = run_git(git, "merge", "feature")
        assert "Merge" in output
        # The new file from feature should now be on main (read through Staged)
        assert state.get("new.py") == b"feature code\n"

    def test_merge_already_up_to_date(self, vkv, state, git):
        commit_files(state, {"a": b"1"}, "init")
        vkv.create_branch("same")
        output = run_git(git, "merge", "same")
        assert "Already up to date" in output

    def test_merge_nonexistent_branch(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        with pytest.raises(TerminalError, match="not found"):
            run_git(git, "merge", "ghost")

    def test_merge_into_self_fails(self, vkv, state, git):
        commit_files(state, {"a": b"1"})
        with pytest.raises(TerminalError, match="cannot merge"):
            run_git(git, "merge", "main")


# =============================================================================
# git add + selective commit
# =============================================================================


class TestAdd:
    def test_add_specific_file_selective_commit(self, vkv, state, git):
        """git add + commit only flushes the tracked file; others stay staged."""
        state["a.py"] = b"aaa"
        state["b.py"] = b"bbb"
        run_git(git, "add", "a.py")
        run_git(git, "commit", "-m", "just a")

        info = vkv.commit_info(vkv.current_commit)
        assert info["message"] == "just a"
        assert info.get("files") == ["a.py"]

        # b.py is still pending in Staged (not committed)
        assert state.is_staged("b.py")

    def test_add_dot_stages_all_changed(self, vkv, state, git):
        """git add . stages all VFS files that changed since last agent commit."""
        state["x.py"] = b"xxx"
        run_git(git, "commit", "-m", "baseline")

        state["x.py"] = b"xxx_v2"
        state["y.py"] = b"yyy"
        run_git(git, "add", ".")
        run_git(git, "commit", "-m", "updated")

        info = vkv.commit_info(vkv.current_commit)
        assert "x.py" in info.get("files", [])
        assert "y.py" in info.get("files", [])

    def test_commit_without_add_has_no_files_annotation(self, vkv, state, git):
        """Without git add, the commit has no files annotation."""
        state["a.py"] = b"aaa"
        run_git(git, "commit", "-m", "everything")

        info = vkv.commit_info(vkv.current_commit)
        assert info["message"] == "everything"
        assert "files" not in info

    def test_status_shows_staged_and_unstaged(self, vkv, state, git):
        """git status distinguishes staged vs unstaged files."""
        state["base.py"] = b"base"
        run_git(git, "commit", "-m", "baseline")

        state["staged.py"] = b"s"
        state["unstaged.py"] = b"u"
        run_git(git, "add", "staged.py")

        output = run_git(git, "status")
        assert "Changes to be committed" in output
        assert "staged.py" in output
        assert "Changes not staged" in output
        assert "unstaged.py" in output

    def test_status_clean_after_commit(self, vkv, state, git):
        """git status shows clean right after a full commit."""
        state["f.py"] = b"x"
        run_git(git, "commit", "-m", "checkpoint")

        output = run_git(git, "status")
        assert "nothing to commit" in output

    def test_commit_nothing_pending_refuses(self, vkv, state, git):
        """git commit with nothing pending refuses like real git."""
        state["f.py"] = b"x"
        run_git(git, "commit", "-m", "first")
        with pytest.raises(TerminalError, match="nothing to commit"):
            run_git(git, "commit", "-m", "empty")

    def test_checkout_with_uncommitted_changes_refuses(self, vkv, state, git):
        """git checkout refuses when there are uncommitted changes."""
        state["a.py"] = b"1"
        run_git(git, "commit", "-m", "init")
        vkv.create_branch("other")
        state["a.py"] = b"modified"
        with pytest.raises(TerminalError, match="local changes"):
            run_git(git, "checkout", "other")

    def test_add_no_args_errors(self, git):
        with pytest.raises(TerminalError, match="nothing specified"):
            run_git(git, "add")


# =============================================================================
# git rm
# =============================================================================


class TestRm:
    def test_rm_removes_file(self, vkv, state, git):
        """git rm deletes a file from the workspace."""
        state["a.py"] = b"aaa"
        run_git(git, "commit", "-m", "add a")

        output = run_git(git, "rm", "a.py")
        assert "rm 'a.py'" in output
        assert "a.py" not in state

    def test_rm_shows_in_status(self, vkv, state, git):
        """After git rm, git status shows the deletion as pending."""
        state["a.py"] = b"aaa"
        run_git(git, "commit", "-m", "add a")

        run_git(git, "rm", "a.py")
        output = run_git(git, "status")
        assert "a.py" in output

    def test_rm_then_commit_persists(self, vkv, state, git):
        """git rm + git commit persists the deletion."""
        state["a.py"] = b"aaa"
        state["b.py"] = b"bbb"
        run_git(git, "commit", "-m", "add both")

        run_git(git, "rm", "a.py")
        run_git(git, "commit", "-m", "remove a")

        output = run_git(git, "log", "--oneline")
        assert "remove a" in output

    def test_rm_nonexistent_errors(self, vkv, state, git):
        """git rm on a file that doesn't exist raises an error."""
        with pytest.raises(TerminalError, match="not found"):
            run_git(git, "rm", "ghost.py")

    def test_rm_no_args_errors(self, git):
        with pytest.raises(TerminalError, match="nothing specified"):
            run_git(git, "rm")


# =============================================================================
# Error cases
# =============================================================================


class TestErrors:
    def test_unknown_subcommand(self, git):
        with pytest.raises(TerminalError, match="not a git command"):
            run_git(git, "stash")

    def test_no_subcommand_shows_usage(self, git):
        output = run_git(git)
        assert "usage:" in output


# =============================================================================
# Pipeline composition via termish execute()
# =============================================================================


class TestMonkeyFSIntegration:
    """Full-stack integration: termish file writes → monkeyfs VirtualFS → kvgit → git CLI."""

    def _setup(self):
        """Create a VersionedKV + VirtualFS connected via monkeyfs."""
        from kvgit import Staged
        from monkeyfs import VirtualFS

        vkv = VersionedKV(Memory())
        state = Staged(vkv)
        vfs = VirtualFS(state)
        git_handler = make_git_handler(vkv, state=state, vfs=vfs)
        return vkv, state, vfs, git_handler

    def test_write_file_commit_and_log(self):
        """Write a file via termish → commit via git → see it in git log."""
        vkv, state, vfs, git_handler = self._setup()
        commands = {"git": git_handler}

        # Write a file via termish (goes through VirtualFS → Staged → kvgit)
        execute("echo 'print(42)' > hello.py", vfs, commands=commands)

        # git commit flushes Staged changes and attaches the message
        output = execute("git commit -m 'initial hello'", vfs, commands=commands)
        assert "initial hello" in output

        # Should show up in git log
        output = execute("git log --oneline", vfs, commands=commands)
        assert "initial hello" in output

    def test_edit_file_and_diff(self):
        """Edit a file → git diff shows the change with clean paths."""
        vkv, state, vfs, git_handler = self._setup()
        commands = {"git": git_handler}

        execute("echo 'version 1' > app.py", vfs, commands=commands)
        execute("git commit -m 'v1'", vfs, commands=commands)

        execute("echo 'version 2' > app.py", vfs, commands=commands)
        execute("git commit -m 'v2'", vfs, commands=commands)

        # Diff should show both versions, with clean paths (no __vfs_ prefix)
        output = execute("git diff HEAD~1", vfs, commands=commands)
        assert "version 1" in output
        assert "version 2" in output
        assert "a/app.py" in output or "b/app.py" in output
        assert "__vfs_" not in output

    def test_show_file_at_older_commit(self):
        """git show HEAD~1:file.py retrieves the old version through VFS."""
        vkv, state, vfs, git_handler = self._setup()
        commands = {"git": git_handler}

        execute("echo 'old content' > data.py", vfs, commands=commands)
        execute("git commit -m 'old'", vfs, commands=commands)

        execute("echo 'new content' > data.py", vfs, commands=commands)
        execute("git commit -m 'new'", vfs, commands=commands)

        output = execute("git show HEAD~1:data.py", vfs, commands=commands)
        assert "old content" in output
        assert "new content" not in output

    def test_branch_edit_and_merge(self):
        """Create branch → edit on branch → merge back to main."""
        vkv, state, vfs, git_handler = self._setup()
        commands = {"git": git_handler}

        execute("echo 'base' > shared.py", vfs, commands=commands)
        execute("git commit -m 'base'", vfs, commands=commands)

        # Create and switch to feature branch
        execute("git checkout -b feature", vfs, commands=commands)
        assert vkv.current_branch == "feature"

        execute("echo 'feature work' > feature.py", vfs, commands=commands)
        execute("git commit -m 'feature work'", vfs, commands=commands)

        # Switch back to main and merge
        execute("git checkout main", vfs, commands=commands)
        assert vkv.current_branch == "main"

        output = execute("git merge feature", vfs, commands=commands)
        assert "Merge" in output

        # feature.py should now exist on main
        assert vfs.exists("feature.py")

    def test_reset_restores_previous_file_state(self):
        """git reset --hard HEAD~1 restores old file content without moving HEAD."""
        vkv, state, vfs, git_handler = self._setup()
        commands = {"git": git_handler}

        execute("echo 'v1' > code.py", vfs, commands=commands)
        execute("git commit -m 'v1'", vfs, commands=commands)

        execute("echo 'v2' > code.py", vfs, commands=commands)
        execute("git commit -m 'v2'", vfs, commands=commands)

        # Reset to v1 — virtual restore, not a HEAD move
        output = execute("git reset --hard HEAD~1", vfs, commands=commands)
        assert "Restored" in output

        # The VFS now has v1 content (restored via Staged writes)
        content = vfs.read("code.py")
        assert b"v1" in content

    def test_full_workflow_with_pipeline(self):
        """End-to-end: write files, commit, pipe git log through grep."""
        vkv, state, vfs, git_handler = self._setup()
        commands = {"git": git_handler}

        execute("echo 'module A' > a.py", vfs, commands=commands)
        execute("git commit -m 'add module A'", vfs, commands=commands)

        execute("echo 'module B' > b.py", vfs, commands=commands)
        execute("git commit -m 'add module B'", vfs, commands=commands)

        execute("echo 'fix for A' > a.py", vfs, commands=commands)
        execute("git commit -m 'fix module A'", vfs, commands=commands)

        # Filter log for commits touching "module A"
        output = execute(
            "git log --oneline | grep 'module A'",
            vfs,
            commands=commands,
        )
        lines = [line for line in output.strip().split("\n") if line]
        assert len(lines) == 2  # "add module A" and "fix module A"
        assert all("module A" in line for line in lines)


class TestPipelineComposition:
    def test_git_log_piped_to_grep(self, vkv, git):
        vkv.commit({"a": b"1"}, info={"message": "add feature X"})
        vkv.commit({"b": b"2"}, info={"message": "fix bug Y"})
        vkv.commit({"c": b"3"}, info={"message": "add feature Z"})

        fs = MemoryFS()
        output = execute(
            "git log --oneline | grep add",
            fs,
            commands={"git": make_git_handler(vkv)},
        )
        lines = [line for line in output.strip().split("\n") if line]
        assert len(lines) == 2
        assert all("add" in line for line in lines)
        assert not any("fix" in line for line in lines)

    def test_git_log_piped_to_wc(self, vkv, git):
        for i in range(4):
            vkv.commit({f"f{i}": b"x"}, info={"message": f"commit {i}"})

        fs = MemoryFS()
        output = execute(
            "git log --oneline | wc -l",
            fs,
            commands={"git": make_git_handler(vkv)},
        )
        # 4 commits + the initial empty commit = 5 (or 4 if initial has no message)
        count = int(output.strip())
        assert count >= 4
