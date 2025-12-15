"""Tests for branch-based snapshot operations with CAS merge."""

import pickle

from agex.state import ConcurrencyError, Versioned, kv
from agex.state.versioned import HEAD_COMMIT


def test_snapshot_and_merge_success():
    """Test that snapshot + merge succeeds when no concurrent modifications."""
    store = kv.Memory()
    state = Versioned(store)

    state.set("a", "value1")
    result = state.snapshot()
    success = state.merge()

    assert result.commit_hash is not None
    assert success is True
    assert state.get("a") == "value1"

    # HEAD should now point to our commit
    head = pickle.loads(store.get(HEAD_COMMIT))
    assert head == state.current_commit


def test_merge_raises_on_diverged_head():
    """Test that merge() raises ConcurrencyError when HEAD diverged."""
    store = kv.Memory()
    state1 = Versioned(store)
    state2 = Versioned(store)

    # Both states start from the same HEAD
    initial_commit = state1.current_commit
    assert state2.current_commit == initial_commit

    # State1 makes a change and merges successfully
    state1.set("a", "value1")
    state1.snapshot()
    state1.merge()

    # State2 tries to merge, but HEAD has changed
    state2.set("b", "value2")
    state2.snapshot()

    try:
        state2.merge()
        assert False, "Expected ConcurrencyError"
    except ConcurrencyError as e:
        assert "Concurrent modification detected" in str(e)
        assert initial_commit in str(e)


def test_merge_abandon_returns_false():
    """Test that merge(on_conflict='abandon') returns False on conflict."""
    store = kv.Memory()
    state1 = Versioned(store)
    state2 = Versioned(store)

    # State1 merges first
    state1.set("a", "value1")
    state1.snapshot()
    state1.merge()

    # State2 tries to merge with abandon strategy
    state2.set("b", "value2")
    state2.snapshot()
    success = state2.merge(on_conflict="abandon")

    assert success is False
    # State2's commits are now orphans


def test_merge_creates_orphan_on_conflict():
    """Test that failed merge leaves orphan data for GC."""
    store = kv.Memory()
    state1 = Versioned(store)
    state2 = Versioned(store)

    # State1 merges
    state1.set("a", "value1")
    state1.snapshot()
    state1.merge()

    # Count metadata keys before state2's failed merge
    meta_keys_before = [k for k in store.keys() if k.startswith("__meta__")]

    # State2 snapshots (creates branch data)
    state2.set("b", "value2")
    state2.snapshot()

    # Count after snapshot (orphan data exists)
    meta_keys_after = [k for k in store.keys() if k.startswith("__meta__")]

    # Should have one more metadata key (state2's branch)
    assert len(meta_keys_after) == len(meta_keys_before) + 1


def test_reset_reloads_from_head():
    """Test that reset() reloads state from current HEAD."""
    store = kv.Memory()
    state1 = Versioned(store)
    state2 = Versioned(store)

    # State1 merges successfully
    state1.set("a", "value1")
    state1.snapshot()
    state1.merge()

    # State2 makes changes but doesn't merge
    state2.set("b", "value2")
    state2.snapshot()

    # Reset state2 to current HEAD
    state2.reset()

    # Should now see state1's data and have updated base
    assert state2.get("a") == "value1"
    assert state2.get("b") is None  # Our unmerged changes are gone
    assert state2.current_commit == state1.current_commit
    assert state2.base_commit == state1.current_commit


def test_retry_after_conflict():
    """Test the retry pattern after ConcurrencyError."""
    store = kv.Memory()
    state1 = Versioned(store)
    state2 = Versioned(store)

    # State1 merges
    state1.set("a", "value1")
    state1.snapshot()
    state1.merge()

    # State2 tries and fails
    state2.set("b", "value2")
    state2.snapshot()
    try:
        state2.merge()
        assert False, "Expected ConcurrencyError"
    except ConcurrencyError:
        pass

    # Reset and retry
    state2.reset()
    assert state2.get("a") == "value1"

    state2.set("b", "value2")
    state2.snapshot()
    success = state2.merge()

    assert success is True

    # Final state has both values
    final = Versioned(store)
    assert final.get("a") == "value1"
    assert final.get("b") == "value2"


def test_multiple_snapshots_before_merge():
    """Test that multiple snapshots on a branch work correctly."""
    store = kv.Memory()
    state = Versioned(store)
    initial = state.current_commit

    # Multiple snapshots without merge
    state.set("a", 1)
    state.snapshot()

    state.set("b", 2)
    state.snapshot()

    state.set("c", 3)
    state.snapshot()
    commit3 = state.current_commit

    # HEAD should still be at initial (no merge yet)
    head = pickle.loads(store.get(HEAD_COMMIT))
    assert head == initial

    # Base should still be initial
    assert state.base_commit == initial

    # Now merge all at once
    success = state.merge()
    assert success is True

    # HEAD updated to final commit
    head = pickle.loads(store.get(HEAD_COMMIT))
    assert head == commit3


def test_no_merge_needed_for_empty_branch():
    """Test that merge() returns True immediately if no commits on branch."""
    store = kv.Memory()
    state = Versioned(store)

    # No snapshot, just merge
    success = state.merge()
    assert success is True


def test_concurrent_workers_with_retries():
    """Test multiple workers making concurrent changes with proper retries."""
    store = kv.Memory()

    # Simulate 3 workers all trying to update
    state1 = Versioned(store)
    state2 = Versioned(store)
    state3 = Versioned(store)

    # Worker 1 succeeds first
    state1.set("worker1", "data1")
    state1.snapshot()
    state1.merge()

    # Workers 2 and 3 snapshot (creates branch data)
    state2.set("worker2", "data2")
    state2.snapshot()

    state3.set("worker3", "data3")
    state3.snapshot()

    # Both will fail merge
    try:
        state2.merge()
        assert False, "Expected ConcurrencyError"
    except ConcurrencyError:
        pass

    try:
        state3.merge()
        assert False, "Expected ConcurrencyError"
    except ConcurrencyError:
        pass

    # Worker 2 retries
    state2.reset()
    state2.set("worker2", "data2")
    state2.snapshot()
    state2.merge()

    # Worker 3 retries
    state3.reset()
    state3.set("worker3", "data3")
    state3.snapshot()
    state3.merge()

    # Verify all data is present
    final = Versioned(store)
    assert final.get("worker1") == "data1"
    assert final.get("worker2") == "data2"
    assert final.get("worker3") == "data3"
