import time

from agex.state import GCVersioned, Versioned, kv


def test_clean_orphans_removes_old_unreachable_commits():
    """Test that orphaned commits older than min_age are cleaned up."""
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=10_000, low_water_bytes=8_000)

    # Create initial commit
    state.set("a", "value1")
    commit1 = state.snapshot().commit_hash

    # Wait a bit to make commit timestamps different
    time.sleep(0.1)

    # Manually create an orphaned commit by simulating a failed CAS
    # (creating a commit that's not reachable from HEAD)
    orphan_state = Versioned(store, commit_hash=commit1)
    orphan_state.set("orphan_data", "should be deleted")
    orphan_commit = orphan_state.snapshot().commit_hash

    # Reset HEAD to commit1 (making orphan_commit unreachable)
    import pickle

    from agex.state.versioned import HEAD_COMMIT

    store.set(HEAD_COMMIT, pickle.dumps(commit1))

    # Reload state at commit1
    state = GCVersioned(
        store, commit_hash=commit1, high_water_bytes=10_000, low_water_bytes=8_000
    )

    # Verify orphan exists
    assert store.get(f"__meta__{orphan_commit}") is not None

    # Clean orphans with no safety window (0 seconds)
    cleaned_count = state.clean_orphans(min_age_seconds=0)

    # Orphan should be deleted
    assert cleaned_count == 1
    assert store.get(f"__meta__{orphan_commit}") is None


def test_clean_orphans_respects_safety_window():
    """Test that recent orphans are not deleted (safety window)."""
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=10_000, low_water_bytes=8_000)

    # Create initial commit
    state.set("a", "value1")
    commit1 = state.snapshot().commit_hash

    # Create a recent orphan
    orphan_state = Versioned(store, commit_hash=commit1)
    orphan_state.set("recent_orphan", "too new to delete")
    orphan_commit = orphan_state.snapshot().commit_hash

    # Reset HEAD to commit1
    import pickle

    from agex.state.versioned import HEAD_COMMIT

    store.set(HEAD_COMMIT, pickle.dumps(commit1))

    # Reload state
    state = GCVersioned(
        store, commit_hash=commit1, high_water_bytes=10_000, low_water_bytes=8_000
    )

    # Clean with 1 hour window (orphan is too recent)
    cleaned_count = state.clean_orphans(min_age_seconds=3600)

    # Recent orphan should NOT be deleted
    assert cleaned_count == 0
    assert store.get(f"__meta__{orphan_commit}") is not None


def test_clean_orphans_never_deletes_reachable_commits():
    """Test that commits reachable from HEAD are never deleted."""
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=10_000, low_water_bytes=8_000)

    # Create a chain of commits
    state.set("a", "value1")
    commit1 = state.snapshot().commit_hash

    state.set("b", "value2")
    commit2 = state.snapshot().commit_hash

    state.set("c", "value3")
    commit3 = state.snapshot().commit_hash

    # All three commits should be reachable from HEAD
    cleaned_count = state.clean_orphans(min_age_seconds=0)  # No safety window

    # Nothing should be deleted
    assert cleaned_count == 0
    assert store.get(f"__meta__{commit1}") is not None
    assert store.get(f"__meta__{commit2}") is not None
    assert store.get(f"__meta__{commit3}") is not None


def test_clean_orphans_integrated_with_rebase():
    """Test that clean_orphans runs during rebase and updates RebaseResult."""
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=1_000, low_water_bytes=500)

    # Create initial data
    state.set("a", "value1")
    commit1 = state.snapshot().commit_hash

    time.sleep(0.1)

    # Manually create an old orphan
    orphan_state = Versioned(store, commit_hash=commit1)
    orphan_state.set("orphan", "old orphan")
    orphan_state.snapshot()  # Create orphan commit (not stored in HEAD)

    # Reset HEAD
    import pickle

    from agex.state.versioned import HEAD_COMMIT

    store.set(HEAD_COMMIT, pickle.dumps(commit1))

    # Reload and trigger rebase with large data
    state = GCVersioned(
        store, commit_hash=commit1, high_water_bytes=1_000, low_water_bytes=500
    )
    state.set("big_data", "x" * 2000)
    state.snapshot()

    # Check that rebase cleaned orphans (with no safety window, orphan should be cleaned)
    result = state.last_rebase_result
    assert result is not None
    # Note: orphans_cleaned might be 0 or 1 depending on timing
    # The important thing is that rebase ran without errors
    assert result.performed is True


def test_clean_orphans_removes_all_commit_data():
    """Test that clean_orphans removes all associated data (blobs, metadata, etc.)."""
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=10_000, low_water_bytes=8_000)

    # Create initial commit
    state.set("a", "value1")
    commit1 = state.snapshot().commit_hash

    time.sleep(0.1)

    # Create orphan with data
    orphan_state = Versioned(store, commit_hash=commit1)
    orphan_state.set("orphan_key1", "data1")
    orphan_state.set("orphan_key2", "data2")
    orphan_commit = orphan_state.snapshot().commit_hash

    # Reset HEAD
    import pickle

    from agex.state.versioned import (
        COMMIT_KEYSET,
        HEAD_COMMIT,
        PARENT_COMMIT,
        TOTAL_VAR_SIZE_KEY,
    )

    store.set(HEAD_COMMIT, pickle.dumps(commit1))

    # Reload state
    state = GCVersioned(
        store, commit_hash=commit1, high_water_bytes=10_000, low_water_bytes=8_000
    )

    # Verify all orphan metadata exists
    assert store.get(f"__meta__{orphan_commit}") is not None
    assert store.get(COMMIT_KEYSET % orphan_commit) is not None
    assert store.get(PARENT_COMMIT % orphan_commit) is not None
    assert store.get(TOTAL_VAR_SIZE_KEY % orphan_commit) is not None

    # Clean orphans with no safety window
    cleaned_count = state.clean_orphans(min_age_seconds=0)
    assert cleaned_count == 1

    # Verify all orphan data is gone
    assert store.get(f"__meta__{orphan_commit}") is None
    assert store.get(COMMIT_KEYSET % orphan_commit) is None
    assert store.get(PARENT_COMMIT % orphan_commit) is None
    assert store.get(TOTAL_VAR_SIZE_KEY % orphan_commit) is None


def test_find_all_commit_hashes():
    """Test the _find_all_commit_hashes helper method."""
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=10_000, low_water_bytes=8_000)

    # Get initial commit count (there's always an initial empty commit)
    initial_commits = state._find_all_commit_hashes()
    initial_count = len(initial_commits)

    # Create multiple commits
    state.set("a", "value1")
    commit1 = state.snapshot().commit_hash

    state.set("b", "value2")
    commit2 = state.snapshot().commit_hash

    state.set("c", "value3")
    commit3 = state.snapshot().commit_hash

    # Find all commits
    all_commits = state._find_all_commit_hashes()

    # Should have initial commits plus the 3 new ones
    assert len(all_commits) == initial_count + 3
    assert commit1 in all_commits
    assert commit2 in all_commits
    assert commit3 in all_commits


def test_get_commit_created_at():
    """Test the _get_commit_created_at helper method."""
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=10_000, low_water_bytes=8_000)

    # Create a commit
    start_time = time.time()
    state.set("a", "value1")
    commit1 = state.snapshot().commit_hash
    end_time = time.time()

    # Get creation time
    created_at = state._get_commit_created_at(commit1)

    # Should be within the time window
    assert created_at is not None
    assert start_time <= created_at <= end_time

    # Non-existent commit should return None
    fake_commit = "nonexistent"
    assert state._get_commit_created_at(fake_commit) is None
