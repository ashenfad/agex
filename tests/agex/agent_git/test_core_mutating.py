"""Tests for VirtualGit mutating operations.

Covers add, rm, commit, reset, create_branch, delete_branch,
checkout, merge, and (critically) the isolation guarantees that were
the original motivation for this refactor: branch / checkout / merge
must NOT disturb non-VFS state (event log, REPL namespace, agent
memory) the way ``vkv.switch_branch`` would.
"""

import pytest
from kvgit import Staged
from kvgit.store import Memory
from kvgit.versioned.kv import VersionedKV

from agex.agent_git import (
    AgentGitError,
    BranchExists,
    BranchNotFound,
    BranchNotMerged,
    Metadata,
    NothingToCommit,
    PathSpecError,
    PendingChanges,
    UnbornBranch,
    VirtualGit,
)

# ---------------------------------------------------------------------------
# Fixtures
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


def head(vg):
    """Convenience: current branch's tip."""
    return vg.head()


def commit_with(vg, files: dict, message: str) -> str:
    """Stage files and commit through VirtualGit."""
    for k, v in files.items():
        vg._state[k] = v
    return vg.commit(message).hash


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_no_args_raises(self, vg):
        with pytest.raises(PathSpecError, match="nothing"):
            vg.add([])

    def test_add_existing_file_after_modification(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg._state["a"] = b"2"
        vg.add(["a"])
        meta = Metadata.load(vg._state)
        assert "a" in meta.index

    def test_add_unchanged_file_is_accepted(self, vg):
        # Real git: ``git add unchanged.txt`` is a no-op but doesn't error.
        commit_with(vg, {"a": b"1"}, "init")
        vg.add(["a"])  # no modification yet — must not raise
        meta = Metadata.load(vg._state)
        assert "a" in meta.index

    def test_add_nonexistent_path_raises(self, vg):
        with pytest.raises(PathSpecError, match="did not match"):
            vg.add(["ghost.py"])

    def test_add_dot_stages_all_modified(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg._state["a"] = b"2"
        vg._state["b"] = b"new"
        vg.add(["."])
        meta = Metadata.load(vg._state)
        assert meta.index == {"a", "b"}

    def test_add_dash_a(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg._state["a"] = b"2"
        vg.add(["-A"])
        assert "a" in Metadata.load(vg._state).index

    def test_add_persists_across_virtualgit_instances(self, vkv, state):
        # The original closure-based ``_tracked`` set was lost between
        # terminal_action invocations.  Metadata persists.
        vg1 = VirtualGit(vkv, state)
        vg1._state["a"] = b"1"
        vg1.commit("init")
        vg1._state["a"] = b"2"
        vg1.add(["a"])

        vg2 = VirtualGit(vkv, state)
        s = vg2.status()
        assert s.staged == ["a"]


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------


class TestRm:
    def test_no_args_raises(self, vg):
        with pytest.raises(PathSpecError, match="nothing"):
            vg.rm([])

    def test_rm_removes_from_working_tree(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.rm(["a"])
        assert "a" not in vg._state

    def test_rm_stages_deletion(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.rm(["a"])
        assert "a" in Metadata.load(vg._state).index

    def test_rm_missing_path_raises(self, vg):
        with pytest.raises(PathSpecError, match="did not match"):
            vg.rm(["nope"])


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------


class TestCommit:
    def test_commit_records_message(self, vg):
        vg._state["a"] = b"1"
        c = vg.commit("hello")
        assert c.message == "hello"

    def test_commit_advances_branch_ref(self, vg):
        vg._state["a"] = b"1"
        c = vg.commit("init")
        assert vg.head() == c.hash
        assert vg.list_branches() == ["main"]

    def test_commit_records_virtual_branch_and_parents(self, vg):
        vg._state["a"] = b"1"
        a = vg.commit("first")
        vg._state["a"] = b"2"
        b = vg.commit("second")

        assert a.virtual_branch == "main"
        assert a.virtual_parents == []
        assert b.virtual_branch == "main"
        assert b.virtual_parents == [a.hash]

    def test_commit_nothing_pending_raises(self, vg):
        vg._state["a"] = b"1"
        vg.commit("first")
        with pytest.raises(NothingToCommit):
            vg.commit("nothing")

    def test_commit_clears_index(self, vg):
        vg._state["a"] = b"1"
        vg._state["b"] = b"1"
        vg.add(["a"])
        vg.commit("partial")
        assert Metadata.load(vg._state).index == set()

    def test_commit_selective_with_index(self, vg):
        vg._state["a"] = b"1"
        vg._state["b"] = b"1"
        vg.add(["a"])
        c = vg.commit("just a")
        assert c.files == ["a"]
        # b should still be visible as unstaged
        assert vg.status().unstaged == ["b"]

    def test_commit_full_when_index_empty(self, vg):
        vg._state["a"] = b"1"
        vg._state["b"] = b"1"
        c = vg.commit("both")
        assert sorted(c.files) == ["a", "b"]

    def test_commit_after_commit_state_flush_still_works(self, vg, state):
        # Simulate what commit_state does between turns: a system commit
        # without a message annotation flushes any pending Staged writes.
        # The next ``git commit -m`` should still produce a virtual
        # commit that captures those changes.
        vg._state["a"] = b"v1"
        vg.commit("init")

        # Edit, then "commit_state" runs (full flush, no message)
        vg._state["a"] = b"v2"
        state.commit(info=None)  # framework system commit
        assert state.has_changes is False  # buffer drained

        # Even though the buffer is empty, the file is "modified" relative
        # to the last virtual branch tip.  Commit must record the change.
        c = vg.commit("real commit")
        assert c.message == "real commit"
        assert c.files == ["a"]

        # And the new branch tip should reflect the v2 content.
        assert vg.show(c.hash, "a") == b"v2"

    def test_commit_records_deletion(self, vg):
        vg._state["a"] = b"1"
        vg._state["b"] = b"1"
        vg.commit("init")
        vg.rm(["a"])
        c = vg.commit("remove a")
        # After commit, a should be absent at the new branch tip.
        snap = vg._state.checkout(c.hash)
        assert "a" not in snap

    def test_commit_index_with_no_real_changes_raises(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        # Add an unchanged path to the index, then try to commit.
        vg.add(["a"])
        with pytest.raises(NothingToCommit):
            vg.commit("nothing real")


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_only_hard(self, vg):
        with pytest.raises(AgentGitError, match="hard"):
            vg.reset("anything", hard=False)

    def test_reset_restores_files(self, vg):
        a = commit_with(vg, {"f": b"v1"}, "v1")
        commit_with(vg, {"f": b"v2"}, "v2")
        vg.reset(a)
        assert vg._state.get("f") == b"v1"

    def test_reset_rewinds_branch_ref(self, vg):
        a = commit_with(vg, {"f": b"v1"}, "v1")
        commit_with(vg, {"f": b"v2"}, "v2")
        vg.reset(a)
        assert vg.head() == a

    def test_reset_clears_index(self, vg):
        a = commit_with(vg, {"f": b"v1"}, "v1")
        vg._state["f"] = b"v2"
        vg.add(["f"])
        vg.reset(a)
        assert Metadata.load(vg._state).index == set()

    def test_reset_does_not_rewind_kvgit_chain(self, vg, vkv):
        # Virtual reset: the agent's branch ref rewinds, but the kvgit
        # physical chain only ever moves forward.  Both pre-reset
        # commits remain in kvgit history (so non-VFS state captured
        # in those commits is still recoverable).
        a = commit_with(vg, {"f": b"v1"}, "v1")
        b = commit_with(vg, {"f": b"v2"}, "v2")
        kvgit_head_before = vkv.current_commit
        history_before = list(vkv.history())

        vg.reset(a)

        # Branch ref rewound to a, but kvgit didn't drop any commits.
        assert vg.head() == a
        history_after = list(vkv.history())
        assert kvgit_head_before in history_after
        assert b in history_after
        # Every pre-reset commit is still reachable in kvgit history.
        assert set(history_before).issubset(set(history_after))


# ---------------------------------------------------------------------------
# create_branch / delete_branch
# ---------------------------------------------------------------------------


class TestCreateBranch:
    def test_create_branch_at_head(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        meta = Metadata.load(vg._state)
        assert "feature" in meta.branches
        assert meta.branches["feature"] == meta.branches["main"]

    def test_create_does_not_switch(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        assert vg.current_branch == "main"

    def test_duplicate_raises(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        with pytest.raises(BranchExists):
            vg.create_branch("feature")

    def test_create_unborn_raises(self, vg):
        with pytest.raises(UnbornBranch):
            vg.create_branch("feature")


class TestDeleteBranch:
    def test_delete_existing(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("temp")
        vg.delete_branch("temp")
        assert "temp" not in vg.list_branches()

    def test_delete_missing_raises(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        with pytest.raises(BranchNotFound):
            vg.delete_branch("ghost")

    def test_delete_current_raises(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        with pytest.raises(AgentGitError, match="currently checked out"):
            vg.delete_branch("main")

    def test_delete_unmerged_raises(self, vg):
        # Branch advances independently of main; safe-delete must refuse.
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        commit_with(vg, {"b": b"new"}, "feature work")
        vg.checkout("main")
        with pytest.raises(BranchNotMerged):
            vg.delete_branch("feature")

    def test_delete_unmerged_with_force(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        commit_with(vg, {"b": b"new"}, "feature work")
        vg.checkout("main")
        vg.delete_branch("feature", force=True)
        assert "feature" not in vg.list_branches()


# ---------------------------------------------------------------------------
# checkout
# ---------------------------------------------------------------------------


class TestCheckout:
    def test_checkout_existing(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        assert vg.current_branch == "feature"

    def test_checkout_no_op_for_current(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.checkout("main")
        assert vg.current_branch == "main"

    def test_checkout_with_create(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.checkout("feature", create=True)
        assert vg.current_branch == "feature"
        assert "feature" in vg.list_branches()

    def test_checkout_create_existing_raises(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        with pytest.raises(BranchExists):
            vg.checkout("feature", create=True)

    def test_checkout_missing_raises(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        with pytest.raises(BranchNotFound):
            vg.checkout("ghost")

    def test_checkout_pending_changes_refused(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg._state["a"] = b"modified"
        with pytest.raises(PendingChanges):
            vg.checkout("feature")

    def test_checkout_with_force_discards_pending(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg._state["a"] = b"modified"
        vg.checkout("feature", force=True)
        # File restored to the branch's tip content
        assert vg._state.get("a") == b"1"

    def test_checkout_applies_file_view(self, vg):
        commit_with(vg, {"a": b"main_a"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        commit_with(vg, {"a": b"feat_a", "b": b"feat_b"}, "feat work")

        vg.checkout("main")
        assert vg._state.get("a") == b"main_a"
        assert "b" not in vg._state

    def test_checkout_clears_index(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg._state["a"] = b"2"
        vg.add(["a"])
        vg.checkout("feature", force=True)
        assert Metadata.load(vg._state).index == set()

    def test_checkout_pending_change_after_commit_state_still_caught(self, vg, state):
        # Pending = content differs from branch tip, regardless of buffer.
        # If commit_state flushed an edit between turns, checkout must
        # still refuse (matching real git's "your changes would be lost").
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")

        # Edit + framework flush — Staged buffer is now drained.
        vg._state["a"] = b"2"
        state.commit(info=None)
        assert state.has_changes is False

        with pytest.raises(PendingChanges):
            vg.checkout("feature")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_self_raises(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        with pytest.raises(AgentGitError, match="itself"):
            vg.merge("main")

    def test_merge_missing_raises(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        with pytest.raises(BranchNotFound):
            vg.merge("ghost")

    def test_merge_already_up_to_date(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("same")
        assert vg.merge("same") is None

    def test_merge_fast_forward(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        commit_with(vg, {"b": b"2"}, "feat work")
        feat_tip = vg.head()
        vg.checkout("main")

        result = vg.merge("feature")
        assert result is not None
        assert result.hash == feat_tip
        assert vg.head() == feat_tip
        assert vg._state.get("b") == b"2"

    def test_merge_true_creates_merge_commit(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        commit_with(vg, {"b": b"feat"}, "feat work")
        feat_tip = vg.head()
        vg.checkout("main")
        commit_with(vg, {"a": b"main_v2"}, "main work")
        main_tip = vg.head()

        result = vg.merge("feature")
        assert result is not None
        assert result.hash != feat_tip
        assert result.hash != main_tip
        assert set(result.virtual_parents) == {main_tip, feat_tip}
        # Both branches' files visible
        assert vg._state.get("a") == b"main_v2"
        assert vg._state.get("b") == b"feat"

    def test_merge_preserves_main_local_only_changes(self, vg):
        # Regression: previously merge took diff(current, source) and
        # overwrote main's independent edit to ``a`` with source's old
        # value.  Proper 3-way diff(base, source) leaves ``a`` alone.
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        commit_with(vg, {"b": b"feat"}, "feat work")  # feature touches b only
        vg.checkout("main")
        commit_with(vg, {"a": b"main_v2"}, "main work")  # main touches a only

        vg.merge("feature")
        assert vg._state.get("a") == b"main_v2"
        assert vg._state.get("b") == b"feat"

    def test_merge_conflict_takes_source(self, vg):
        # Both branches changed ``a`` differently — v1 says source wins.
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        commit_with(vg, {"a": b"from_feat"}, "feat work")
        vg.checkout("main")
        commit_with(vg, {"a": b"from_main"}, "main work")
        vg.merge("feature")
        assert vg._state.get("a") == b"from_feat"

    def test_merge_pending_changes_refused(self, vg):
        commit_with(vg, {"a": b"1"}, "init")
        vg.create_branch("feature")
        vg.checkout("feature")
        commit_with(vg, {"b": b"feat"}, "feat work")
        vg.checkout("main")
        vg._state["a"] = b"dirty"
        with pytest.raises(PendingChanges):
            vg.merge("feature")


# ---------------------------------------------------------------------------
# Isolation: non-VFS state is preserved across branch operations
# ---------------------------------------------------------------------------


class TestIsolation:
    """The original bug: branch / checkout / merge MUST NOT touch
    non-VFS state.  These tests stand in for the framework's event log,
    REPL namespace, and agent memory by writing arbitrary non-VFS keys
    and asserting they survive every branch operation.
    """

    def _seed_substrate(self, state):
        """Populate non-VFS keys that mimic framework state.

        Uses raw kvgit (no VFS), so VirtualGit's ``_is_visible`` filter
        treats every key here as "visible" when no vfs is configured —
        which is why these tests pass an explicit ``vfs`` to
        VirtualGit.  We use a stub VFS that only treats ``__vfs_*``
        keys as visible.
        """
        state["__event_log__/0"] = b"event-A"
        state["__event_log__/1"] = b"event-B"
        state["repl/x"] = b"some-namespace-value"

    def _make_vg_with_vfs_filter(self, vkv, state):
        """Build a VirtualGit with a stub VFS so non-VFS keys are
        filtered out of the visible set (mirroring production wiring).
        """

        class _StubVFS:
            PREFIX = "__vfs_"
            METADATA_KEY = "__vfs_metadata__"
            CWD_KEY = "__vfs_cwd__"

            def _is_vfs_key(self, key):
                return key.startswith(self.PREFIX)

            def _encode_path(self, path):
                # Trivial encoding: prefix-and-strip.  The filter in
                # _is_visible only cares about the prefix, so this
                # round-trips cleanly for tests.
                return self.PREFIX + path

            def _decode_path(self, key):
                return key[len(self.PREFIX) :]

        return VirtualGit(vkv, state, vfs=_StubVFS())

    def test_checkout_preserves_event_log(self, vkv, state):
        vg = self._make_vg_with_vfs_filter(vkv, state)
        # Make a VFS-encoded file so we have something to commit.
        vg._state["__vfs_a"] = b"1"
        vg.commit("init")
        vg.create_branch("feature")

        # Seed non-VFS state AFTER the first commit so it's unequivocally
        # the substrate's own state, not a file.
        self._seed_substrate(state)

        vg.checkout("feature")
        assert state.get("__event_log__/0") == b"event-A"
        assert state.get("__event_log__/1") == b"event-B"
        assert state.get("repl/x") == b"some-namespace-value"

    def test_checkout_preserves_kvgit_branch(self, vkv, state):
        # The big bug: ``vkv.switch_branch`` was being called by
        # ``_git_checkout``.  The fix means kvgit's own branch must
        # NOT change when the agent does ``git checkout``.
        vg = self._make_vg_with_vfs_filter(vkv, state)
        vg._state["__vfs_a"] = b"1"
        vg.commit("init")
        vg.create_branch("feature")

        kvgit_branch_before = vkv.current_branch
        vg.checkout("feature")
        assert vkv.current_branch == kvgit_branch_before

    def test_create_branch_does_not_create_kvgit_branch(self, vkv, state):
        vg = self._make_vg_with_vfs_filter(vkv, state)
        vg._state["__vfs_a"] = b"1"
        vg.commit("init")

        kvgit_branches_before = set(vkv.list_branches())
        vg.create_branch("experiment")
        # kvgit's branch list must be unchanged
        assert set(vkv.list_branches()) == kvgit_branches_before
        # but the virtual branch shows up
        assert "experiment" in vg.list_branches()

    def test_delete_branch_does_not_delete_kvgit_branch(self, vkv, state):
        vg = self._make_vg_with_vfs_filter(vkv, state)
        vg._state["__vfs_a"] = b"1"
        vg.commit("init")
        vg.create_branch("temp")

        kvgit_branches_before = set(vkv.list_branches())
        vg.delete_branch("temp")
        assert set(vkv.list_branches()) == kvgit_branches_before

    def test_merge_preserves_event_log(self, vkv, state):
        vg = self._make_vg_with_vfs_filter(vkv, state)
        vg._state["__vfs_a"] = b"1"
        vg.commit("init")
        vg.create_branch("feature")
        vg.checkout("feature")
        vg._state["__vfs_b"] = b"feat"
        vg.commit("feat work")
        vg.checkout("main")

        self._seed_substrate(state)
        vg.merge("feature")

        assert state.get("__event_log__/0") == b"event-A"
        assert state.get("repl/x") == b"some-namespace-value"
