"""Tests for VirtualGit read-only operations.

Covers status, log, diff, show, list_branches, current_branch,
resolve_ref — all the methods that don't mutate metadata or files.
"""

import pytest
from kvgit import Staged
from kvgit.store import Memory
from kvgit.versioned.kv import VersionedKV

from agex.agent_git import (
    AgentCommit,
    InvalidRef,
    Metadata,
    Status,
    VirtualGit,
)

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
def vg(vkv, state):
    return VirtualGit(vkv, state)


def commit_agent(state, files: dict, message: str, *, branch="main", parents=None):
    """Make an agent commit that updates ``files`` and returns its hash."""
    for k, v in files.items():
        state[k] = v
    info = {
        "message": message,
        "virtual_branch": branch,
        "virtual_parents": list(parents or []),
    }
    return state.commit(info=info).commit


def system_commit(state, key, value):
    """Make a framework-style commit (no message, no virtual_branch)."""
    state[key] = value
    return state.commit(info=None).commit


def set_meta(state, **fields):
    """Replace metadata in one shot.  Stores immediately (does not commit)."""
    Metadata(**fields).save(state)


# ---------------------------------------------------------------------------
# Branch state
# ---------------------------------------------------------------------------


class TestBranchState:
    def test_default_current_branch(self, vg):
        assert vg.current_branch == "main"

    def test_no_branches_listed_for_unborn(self, vg):
        assert vg.list_branches() == []

    def test_list_branches_sorted(self, state, vg):
        set_meta(state, branches={"zeta": "abc", "alpha": "def"})
        assert vg.list_branches() == ["alpha", "zeta"]

    def test_head_unborn(self, vg):
        assert vg.head() is None

    def test_head_set(self, state, vg):
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h})
        assert vg.head() == h


# ---------------------------------------------------------------------------
# resolve_ref delegation
# ---------------------------------------------------------------------------


class TestResolveRef:
    def test_head_resolves(self, state, vg):
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h})
        assert vg.resolve_ref("HEAD") == h

    def test_invalid_raises(self, vg):
        with pytest.raises(InvalidRef):
            vg.resolve_ref("nope")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_clean_unborn_branch(self, vg):
        s = vg.status()
        assert s.branch == "main"
        assert s.staged == []
        assert s.unstaged == []
        assert s.is_clean

    def test_unborn_with_writes_shows_unstaged(self, state, vg):
        # Working tree has a file but no commits → unstaged add.
        state["a"] = b"1"
        s = vg.status()
        assert s.unstaged == ["a"]
        assert s.staged == []
        assert not s.is_clean

    def test_clean_after_commit(self, state, vg):
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h})
        s = vg.status()
        assert s.is_clean

    def test_unstaged_modification(self, state, vg):
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h})
        state["a"] = b"2"
        s = vg.status()
        assert s.unstaged == ["a"]
        assert s.staged == []

    def test_staged_modification(self, state, vg):
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h}, index={"a"})
        state["a"] = b"2"
        s = vg.status()
        assert s.staged == ["a"]
        assert s.unstaged == []

    def test_split_staged_and_unstaged(self, state, vg):
        h = commit_agent(state, {"a": b"1", "b": b"1"}, "init")
        set_meta(state, branches={"main": h}, index={"a"})
        state["a"] = b"2"
        state["b"] = b"2"
        s = vg.status()
        assert s.staged == ["a"]
        assert s.unstaged == ["b"]

    def test_deleted_file_appears_modified(self, state, vg):
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h})
        del state["a"]
        s = vg.status()
        assert s.unstaged == ["a"]

    def test_metadata_key_not_in_modified(self, state, vg):
        # Saving metadata should not show up as a modified file —
        # METADATA_KEY is filtered by _is_visible.
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h})
        # Metadata save happens inside set_meta; status should be clean.
        assert vg.status().is_clean


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


class TestLog:
    def test_unborn_returns_empty(self, vg):
        assert vg.log() == []

    def test_single_commit(self, state, vg):
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h})
        log = vg.log()
        assert len(log) == 1
        assert log[0].hash == h
        assert log[0].message == "init"
        assert log[0].virtual_branch == "main"
        assert log[0].virtual_parents == []

    def test_linear_chain_walks_virtual_ancestry(self, state, vg):
        a = commit_agent(state, {"a": b"1"}, "first")
        b = commit_agent(state, {"a": b"2"}, "second", parents=[a])
        c = commit_agent(state, {"a": b"3"}, "third", parents=[b])
        set_meta(state, branches={"main": c})
        log = vg.log()
        assert [e.message for e in log] == ["third", "second", "first"]
        assert [e.hash for e in log] == [c, b, a]

    def test_skips_system_commits(self, state, vg):
        a = commit_agent(state, {"a": b"1"}, "first")
        system_commit(state, "_sys", b"x")
        c = commit_agent(state, {"a": b"2"}, "second", parents=[a])
        set_meta(state, branches={"main": c})
        log = vg.log()
        assert [e.message for e in log] == ["second", "first"]

    def test_max_count(self, state, vg):
        a = commit_agent(state, {"a": b"1"}, "first")
        b = commit_agent(state, {"a": b"2"}, "second", parents=[a])
        c = commit_agent(state, {"a": b"3"}, "third", parents=[b])
        set_meta(state, branches={"main": c})
        log = vg.log(max_count=2)
        assert [e.message for e in log] == ["third", "second"]

    def test_path_filter(self, state, vg):
        a = commit_agent(state, {"a": b"1"}, "touched a")
        b = commit_agent(state, {"b": b"2"}, "touched b", parents=[a])
        set_meta(state, branches={"main": b})
        log = vg.log(path="a")
        assert [e.message for e in log] == ["touched a"]

    def test_log_includes_files_when_annotated(self, state, vg):
        # Manually craft a commit info with the "files" annotation.
        state["a"] = b"1"
        result = state.commit(
            info={
                "message": "selective",
                "files": ["a"],
                "virtual_branch": "main",
                "virtual_parents": [],
            }
        )
        h = result.commit
        set_meta(state, branches={"main": h})
        log = vg.log()
        assert log[0].files == ["a"]


# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------


class TestShow:
    def test_show_at_commit(self, state, vg):
        h = commit_agent(state, {"a": b"hello"}, "init")
        set_meta(state, branches={"main": h})
        assert vg.show(h, "a") == b"hello"

    def test_show_at_older_commit(self, state, vg):
        a = commit_agent(state, {"f": b"old"}, "v1")
        b = commit_agent(state, {"f": b"new"}, "v2", parents=[a])
        set_meta(state, branches={"main": b})
        assert vg.show(a, "f") == b"old"
        assert vg.show(b, "f") == b"new"

    def test_show_missing_path_raises(self, state, vg):
        h = commit_agent(state, {"a": b"1"}, "init")
        set_meta(state, branches={"main": h})
        with pytest.raises(FileNotFoundError, match="not found"):
            vg.show(h, "nope")

    def test_show_invalid_commit_raises(self, vg):
        with pytest.raises(InvalidRef, match="not found"):
            vg.show("0" * 40, "any")


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_two_commits(self, state, vg):
        a = commit_agent(state, {"f": b"hello\n"}, "v1")
        b = commit_agent(state, {"f": b"world\n"}, "v2", parents=[a])
        set_meta(state, branches={"main": b})
        out = vg.diff(a, b)
        assert "hello" in out
        assert "world" in out
        assert "-hello" in out
        assert "+world" in out
        assert "a/f" in out
        assert "b/f" in out

    def test_diff_default_head_vs_working(self, state, vg):
        h = commit_agent(state, {"f": b"old\n"}, "v1")
        set_meta(state, branches={"main": h})
        state["f"] = b"new\n"
        out = vg.diff()
        assert "old" in out
        assert "new" in out

    def test_diff_clean_returns_empty(self, state, vg):
        h = commit_agent(state, {"f": b"x"}, "v1")
        set_meta(state, branches={"main": h})
        assert vg.diff() == ""

    def test_diff_unborn_returns_empty(self, vg):
        assert vg.diff() == ""

    def test_diff_added_file(self, state, vg):
        a = commit_agent(state, {"a": b"1"}, "v1")
        b = commit_agent(state, {"a": b"1", "b": b"new\n"}, "v2", parents=[a])
        set_meta(state, branches={"main": b})
        out = vg.diff(a, b)
        assert "b/b" in out
        assert "new" in out

    def test_diff_path_filter(self, state, vg):
        a = commit_agent(state, {"a": b"1\n", "b": b"1\n"}, "v1")
        b = commit_agent(state, {"a": b"2\n", "b": b"2\n"}, "v2", parents=[a])
        set_meta(state, branches={"main": b})
        out = vg.diff(a, b, path="a")
        assert "a/a" in out
        assert "b/b" not in out

    def test_diff_binary_summary(self, state, vg):
        a = commit_agent(state, {"f": b"\x00\x01\x02\x03"}, "v1")
        b = commit_agent(state, {"f": b"\x00\x05\x06\x07"}, "v2", parents=[a])
        set_meta(state, branches={"main": b})
        out = vg.diff(a, b)
        assert "Binary files" in out

    def test_diff_one_arg_means_ref_vs_working(self, state, vg):
        # vg.diff(a, None) → diff a vs working tree.  After staging
        # a change but before committing, this should show old→new.
        a = commit_agent(state, {"f": b"old\n"}, "v1")
        set_meta(state, branches={"main": a})
        state["f"] = b"new\n"
        out = vg.diff(a, None)
        assert "old" in out
        assert "new" in out


# ---------------------------------------------------------------------------
# Result type sanity
# ---------------------------------------------------------------------------


class TestResultTypes:
    def test_agent_commit_short_hash(self):
        c = AgentCommit(
            hash="abcdef1234567890",
            message="x",
            virtual_branch="main",
            virtual_parents=[],
            files=None,
        )
        assert c.short_hash == "abcdef1"

    def test_status_is_clean(self):
        assert Status(branch="main", staged=[], unstaged=[]).is_clean
        assert not Status(branch="main", staged=["a"], unstaged=[]).is_clean
        assert not Status(branch="main", staged=[], unstaged=["a"]).is_clean
