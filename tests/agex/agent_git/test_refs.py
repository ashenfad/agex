"""Tests for agex.agent_git.refs."""

import pytest
from kvgit import Staged
from kvgit.store import Memory
from kvgit.versioned.kv import VersionedKV

from agex.agent_git.metadata import Metadata
from agex.agent_git.refs import (
    InvalidRef,
    all_agent_commits,
    all_ancestors,
    is_agent_commit,
    merge_base,
    resolve_ref,
    virtual_parents,
    walk_virtual_ancestry,
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


def _agent_commit(
    state, key, value, message, *, virtual_branch="main", virtual_parents=None
):
    """Make an agent commit with virtual_branch / virtual_parents annotations.

    Returns the resulting commit hash.
    """
    state[key] = value
    info = {
        "message": message,
        "virtual_branch": virtual_branch,
        "virtual_parents": list(virtual_parents or []),
    }
    result = state.commit(info=info)
    return result.commit


def _system_commit(state, key, value):
    """Make a framework-style commit (no message in info)."""
    state[key] = value
    return state.commit(info=None).commit


# ---------------------------------------------------------------------------
# is_agent_commit / all_agent_commits
# ---------------------------------------------------------------------------


class TestAgentCommitIdentity:
    def test_messaged_commit_is_agent(self, state, vkv):
        h = _agent_commit(state, "a", b"1", "hello")
        assert is_agent_commit(vkv, h)

    def test_unmessaged_commit_is_not_agent(self, state, vkv):
        h = _system_commit(state, "a", b"1")
        assert not is_agent_commit(vkv, h)

    def test_all_agent_commits_filters_system(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        _system_commit(state, "b", b"2")
        c = _agent_commit(state, "c", b"3", "second")

        all_ag = all_agent_commits(vkv)
        assert set(all_ag) == {a, c}
        # Newest-first ordering preserved from vkv.history()
        assert all_ag.index(c) < all_ag.index(a)


# ---------------------------------------------------------------------------
# virtual_parents
# ---------------------------------------------------------------------------


class TestVirtualParents:
    def test_no_recorded_parents(self, state, vkv):
        h = _agent_commit(state, "a", b"1", "init")
        assert virtual_parents(vkv, h) == []

    def test_single_parent(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        b = _agent_commit(state, "a", b"2", "second", virtual_parents=[a])
        assert virtual_parents(vkv, b) == [a]

    def test_merge_two_parents(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        b = _agent_commit(state, "b", b"2", "second", virtual_parents=[a])
        merge = _agent_commit(state, "c", b"3", "merge", virtual_parents=[b, a])
        assert virtual_parents(vkv, merge) == [b, a]

    def test_system_commit_has_none(self, state, vkv):
        h = _system_commit(state, "a", b"1")
        assert virtual_parents(vkv, h) == []


# ---------------------------------------------------------------------------
# walk_virtual_ancestry
# ---------------------------------------------------------------------------


class TestWalkAncestry:
    def test_unborn_yields_nothing(self, vkv):
        assert list(walk_virtual_ancestry(vkv, None)) == []

    def test_root_only(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "init")
        assert list(walk_virtual_ancestry(vkv, a)) == [a]

    def test_linear_chain(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        b = _agent_commit(state, "a", b"2", "second", virtual_parents=[a])
        c = _agent_commit(state, "a", b"3", "third", virtual_parents=[b])
        assert list(walk_virtual_ancestry(vkv, c)) == [c, b, a]

    def test_skips_system_commits(self, state, vkv):
        # A system commit between two agent commits is invisible to the
        # virtual walk because virtual_parents points around it.
        a = _agent_commit(state, "a", b"1", "first")
        _system_commit(state, "b", b"sys")
        c = _agent_commit(state, "a", b"2", "third", virtual_parents=[a])
        assert list(walk_virtual_ancestry(vkv, c)) == [c, a]

    def test_first_parent_only_through_merge(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        b = _agent_commit(state, "b", b"1", "branch", virtual_parents=[a])
        c = _agent_commit(state, "a", b"2", "main2", virtual_parents=[a])
        merge = _agent_commit(state, "c", b"3", "merge", virtual_parents=[c, b])
        # First-parent walk follows main, ignores the merged-in branch tip
        assert list(walk_virtual_ancestry(vkv, merge)) == [merge, c, a]

    def test_cycle_does_not_loop_forever(self, state, vkv, monkeypatch):
        # Synthesise a cycle by stubbing virtual_parents.  Real stores
        # cannot produce one (content addressing forbids it), but a
        # corrupt store shouldn't deadlock the CLI.
        a = _agent_commit(state, "a", b"1", "first")
        b = _agent_commit(state, "a", b"2", "second", virtual_parents=[a])

        from agex.agent_git import refs as refs_mod

        def fake_parents(_vkv, h):
            # a → b → a → ...
            return [b] if h == a else [a]

        monkeypatch.setattr(refs_mod, "virtual_parents", fake_parents)
        result = list(walk_virtual_ancestry(vkv, a))
        # Should terminate, visiting each at most once
        assert set(result) == {a, b}


# ---------------------------------------------------------------------------
# resolve_ref
# ---------------------------------------------------------------------------


class TestResolveRef:
    def test_empty_ref_raises(self, vkv):
        with pytest.raises(InvalidRef, match="empty"):
            resolve_ref("", vkv, Metadata())

    def test_head_unborn_raises(self, vkv):
        with pytest.raises(InvalidRef, match="unborn"):
            resolve_ref("HEAD", vkv, Metadata())

    def test_head_resolves_to_branch_tip(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        meta = Metadata(branches={"main": a})
        assert resolve_ref("HEAD", vkv, meta) == a

    def test_head_tilde_zero_is_head(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        meta = Metadata(branches={"main": a})
        assert resolve_ref("HEAD~0", vkv, meta) == a

    def test_head_tilde_walks_virtual_ancestry(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        b = _agent_commit(state, "a", b"2", "second", virtual_parents=[a])
        c = _agent_commit(state, "a", b"3", "third", virtual_parents=[b])
        meta = Metadata(branches={"main": c})
        assert resolve_ref("HEAD~1", vkv, meta) == b
        assert resolve_ref("HEAD~2", vkv, meta) == a

    def test_head_tilde_too_deep_raises(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        meta = Metadata(branches={"main": a})
        with pytest.raises(InvalidRef, match="beyond"):
            resolve_ref("HEAD~5", vkv, meta)

    def test_head_tilde_invalid_int_raises(self, vkv):
        with pytest.raises(InvalidRef, match="invalid ref"):
            resolve_ref("HEAD~abc", vkv, Metadata())

    def test_head_tilde_negative_raises(self, vkv):
        with pytest.raises(InvalidRef, match="invalid ref"):
            resolve_ref("HEAD~-1", vkv, Metadata())

    def test_branch_name_resolves(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "main")
        b = _agent_commit(
            state, "a", b"2", "feat", virtual_branch="feature", virtual_parents=[a]
        )
        meta = Metadata(branches={"main": a, "feature": b})
        assert resolve_ref("main", vkv, meta) == a
        assert resolve_ref("feature", vkv, meta) == b

    def test_branch_takes_precedence_over_hash_prefix(self, state, vkv):
        # A pathologically-named branch shadowing a hash prefix wins.
        a = _agent_commit(state, "a", b"1", "first")
        # Use a branch named with a prefix that also matches the hash —
        # we'll just verify name-lookup happens first by giving the
        # branch a different commit.
        b = _agent_commit(state, "a", b"2", "second", virtual_parents=[a])
        meta = Metadata(branches={"main": a, a[:7]: b})
        assert resolve_ref(a[:7], vkv, meta) == b  # branch wins, not hash

    def test_hash_prefix_matches_agent_commit(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        meta = Metadata(branches={"main": a})
        assert resolve_ref(a[:7], vkv, meta) == a
        assert resolve_ref(a[:10], vkv, meta) == a

    def test_hash_prefix_works_across_branches(self, state, vkv):
        # A commit on a non-current branch is still addressable by hash.
        a = _agent_commit(state, "a", b"1", "main")
        b = _agent_commit(
            state, "a", b"2", "feat", virtual_branch="feature", virtual_parents=[a]
        )
        meta = Metadata(branches={"main": a})  # feature not in branches
        assert resolve_ref(b[:7], vkv, meta) == b

    def test_hash_prefix_does_not_match_system_commits(self, state, vkv):
        sys_h = _system_commit(state, "a", b"1")
        with pytest.raises(InvalidRef, match="not a valid ref"):
            resolve_ref(sys_h[:7], vkv, Metadata())

    def test_hash_prefix_too_short_raises(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        meta = Metadata(branches={"main": a})
        with pytest.raises(InvalidRef, match="not a valid ref"):
            resolve_ref(a[:6], vkv, meta)

    def test_unknown_ref_raises(self, vkv):
        with pytest.raises(InvalidRef, match="not a valid ref"):
            resolve_ref("nope", vkv, Metadata())


# ---------------------------------------------------------------------------
# all_ancestors / merge_base
# ---------------------------------------------------------------------------


class TestAllAncestors:
    def test_unborn(self, vkv):
        assert all_ancestors(vkv, None) == set()

    def test_includes_self(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        assert all_ancestors(vkv, a) == {a}

    def test_walks_both_parents_through_merge(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        b = _agent_commit(state, "b", b"1", "branch", virtual_parents=[a])
        c = _agent_commit(state, "a", b"2", "main2", virtual_parents=[a])
        merge = _agent_commit(state, "c", b"3", "merge", virtual_parents=[c, b])
        # Unlike walk_virtual_ancestry (first-parent only), the DAG
        # walk reaches b through the merge's second parent.
        assert all_ancestors(vkv, merge) == {merge, c, b, a}


class TestMergeBase:
    def test_none_for_unborn_inputs(self, vkv):
        assert merge_base(vkv, None, "abc") is None
        assert merge_base(vkv, "abc", None) is None

    def test_self_is_own_base(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        assert merge_base(vkv, a, a) == a

    def test_ancestor_is_own_base(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        b = _agent_commit(state, "a", b"2", "second", virtual_parents=[a])
        assert merge_base(vkv, a, b) == a
        assert merge_base(vkv, b, a) == a

    def test_finds_lca_for_diverged_branches(self, state, vkv):
        a = _agent_commit(state, "a", b"1", "first")
        feature_tip = _agent_commit(state, "b", b"1", "feat", virtual_parents=[a])
        main_tip = _agent_commit(state, "a", b"2", "main2", virtual_parents=[a])
        assert merge_base(vkv, main_tip, feature_tip) == a

    def test_unrelated_histories_return_none(self, state, vkv):
        # Two roots with no shared ancestor: simulate by creating
        # two commits with no virtual_parents (each is its own root).
        a = _agent_commit(state, "a", b"1", "rootA")
        b = _agent_commit(state, "b", b"1", "rootB")  # no parents
        assert merge_base(vkv, a, b) is None
