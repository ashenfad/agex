"""Tests for agex.agent_git.metadata."""

import pytest
from kvgit import Staged
from kvgit.store import Memory
from kvgit.versioned.kv import VersionedKV

from agex.agent_git.metadata import DEFAULT_BRANCH, METADATA_KEY, Metadata


@pytest.fixture
def state():
    """Fresh Staged over an in-memory kvgit store."""
    return Staged(VersionedKV(Memory()))


# ---------------------------------------------------------------------------
# In-memory shape
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_fresh_metadata(self):
        m = Metadata()
        assert m.current == DEFAULT_BRANCH
        assert m.branches == {}
        assert m.index == set()

    def test_head_unborn(self):
        assert Metadata().head is None

    def test_head_present(self):
        m = Metadata(current="feature", branches={"feature": "abc"})
        assert m.head == "abc"

    def test_head_branch_missing(self):
        # current points at a branch with no entry — unborn, not an error.
        m = Metadata(current="ghost", branches={"main": "abc"})
        assert m.head is None


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_empty_store(self, state):
        m = Metadata.load(state)
        assert m.current == DEFAULT_BRANCH
        assert m.branches == {}
        assert m.index == set()

    def test_load_after_save(self, state):
        Metadata(
            current="feature",
            branches={"main": "abc", "feature": "def"},
            index={"__vfs_x", "__vfs_y"},
        ).save(state)

        loaded = Metadata.load(state)
        assert loaded.current == "feature"
        assert loaded.branches == {"main": "abc", "feature": "def"}
        assert loaded.index == {"__vfs_x", "__vfs_y"}

    def test_load_partial_blob_defaults_safely(self, state):
        # Older or degraded blob shouldn't crash a load.
        state[METADATA_KEY] = {"current": "main"}
        loaded = Metadata.load(state)
        assert loaded.current == "main"
        assert loaded.branches == {}
        assert loaded.index == set()

    def test_load_none_fields_defaults_safely(self, state):
        state[METADATA_KEY] = {
            "current": "main",
            "branches": None,
            "index": None,
        }
        loaded = Metadata.load(state)
        assert loaded.branches == {}
        assert loaded.index == set()


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_writes_at_reserved_key(self, state):
        Metadata(branches={"main": "abc"}).save(state)
        assert METADATA_KEY in state

    def test_save_serialises_index_sorted(self, state):
        Metadata(index={"b", "a", "c"}).save(state)
        raw = state.get(METADATA_KEY)
        assert raw["index"] == ["a", "b", "c"]

    def test_save_round_trip_through_kvgit_commit(self, state):
        # Sanity: a real kvgit commit + load preserves the blob.
        Metadata(
            current="feature",
            branches={"feature": "abc"},
            index={"k"},
        ).save(state)
        state.commit(info={"message": "test"})

        loaded = Metadata.load(state)
        assert loaded.current == "feature"
        assert loaded.branches == {"feature": "abc"}
        assert loaded.index == {"k"}

    def test_save_overwrites_previous(self, state):
        Metadata(branches={"main": "abc"}).save(state)
        Metadata(branches={"main": "xyz"}).save(state)
        assert Metadata.load(state).branches == {"main": "xyz"}

    def test_save_decouples_input_dict(self, state):
        # Mutating the original dict after save() must not leak back.
        branches = {"main": "abc"}
        m = Metadata(branches=branches)
        m.save(state)
        branches["main"] = "MUTATED"

        loaded = Metadata.load(state)
        assert loaded.branches == {"main": "abc"}


# ---------------------------------------------------------------------------
# Isolation contracts
# ---------------------------------------------------------------------------


class TestKeyIsolation:
    def test_metadata_key_is_not_a_vfs_key(self):
        # The agent-git metadata key MUST NOT start with monkeyfs's
        # ``__vfs_`` prefix or VirtualFS would treat it as a file.
        from monkeyfs import VirtualFS

        assert not METADATA_KEY.startswith(VirtualFS.PREFIX)
