from agex.state import kv
from agex.state.versioned import Versioned


def test_versioned_get_set_remove():
    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    assert state.get("a") == 1
    state.remove("a")
    assert state.get("a") is None


def test_versioned_snapshot():
    import pickle

    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    h1 = state.snapshot().commit_hash
    # KV store now returns bytes, so we need to deserialize
    serialized_value = state.long_term.get(f"{h1}:a")
    assert serialized_value is not None
    assert pickle.loads(serialized_value) == 1

    state.set("a", 2)
    h2 = state.snapshot().commit_hash
    serialized_value2 = state.long_term.get(f"{h2}:a")
    assert serialized_value2 is not None
    assert pickle.loads(serialized_value2) == 2

    # Check that h1 is still there
    serialized_value1_check = state.long_term.get(f"{h1}:a")
    assert serialized_value1_check is not None
    assert pickle.loads(serialized_value1_check) == 1


def test_versioned_history():
    store = kv.Memory()
    state = Versioned(store)

    # Capture the initial commit hash
    h0 = state.current_commit

    state.set("a", 1)
    h1 = state.snapshot().commit_hash
    state.set("b", 2)
    h2 = state.snapshot().commit_hash

    history = list(state.history())
    # History now includes the initial commit
    assert history == [h2, h1, h0]


def test_snapshot_creates_diff_keys():
    store = Versioned(kv.Memory())
    store.set("x", 1)
    store.set("y", 2)
    store.set("__internal__", "should be ignored")
    commit_hash = store.snapshot().commit_hash

    commit_state = store.checkout(commit_hash)  # type: ignore
    diff_keys = commit_state.get("__diff_keys__")  # type: ignore
    assert diff_keys == ("x", "y")


def test_diffs():
    store = Versioned(kv.Memory())
    store.set("a", 100)
    store.snapshot()

    # First set of changes
    store.set("x", 1)
    store.set("y", 2)
    store.set("__event_log__", ["event1"])
    commit1 = store.snapshot().commit_hash

    # Second set of changes
    store.set("y", 3)
    store.set("z", 4)
    store.set("__event_log__", ["event2"])
    commit2 = store.snapshot().commit_hash

    # Check changes for commit 1
    state_changes = store.diffs(commit1)
    assert state_changes == {"x": 1, "y": 2}
    assert store.checkout(commit1).get("__event_log__") == ["event1"]  # type: ignore

    # Check changes for commit 2
    state_changes_2 = store.diffs(commit2)
    assert state_changes_2 == {"y": 3, "z": 4}

    # Check changes for the most recent commit (default)
    state_changes_3 = store.diffs()
    assert state_changes_3 == state_changes_2
    assert store.get("__event_log__") == ["event2"]


def test_snapshot_on_empty_live_preserves_initial_commit():
    store = Versioned(kv.Memory())
    # Versioned now always has an initial commit hash (like Git's empty state)
    initial_commit = store.current_commit
    assert initial_commit is not None

    # Snapshot on empty state returns the same initial commit
    commit1 = store.snapshot().commit_hash
    assert commit1 == initial_commit

    # Adding data creates a new commit
    store.set("a", 1)
    commit2 = store.snapshot().commit_hash
    assert commit2 is not None
    assert commit2 != initial_commit

    # Snapshot without changes returns the same commit
    commit3 = store.snapshot().commit_hash
    assert commit2 == commit3


def test_mutation_detection_prevents_data_loss():
    """Test that side-effect mutations to retrieved objects are detected and preserved."""
    store = kv.Memory()
    state = Versioned(store)

    # Set up initial state with a mutable object
    original_list = [1, 2, 3]
    state.set("my_list", original_list)
    commit1 = state.snapshot().commit_hash

    # Retrieve the object and mutate it in-place (the sneaky bug!)
    retrieved_list = state.get("my_list")
    retrieved_list.append(4)  # This is a side-effect mutation

    # The mutation should be detected during snapshot
    commit2 = state.snapshot().commit_hash

    # Verify the mutation was preserved
    assert state.get("my_list") == [1, 2, 3, 4]

    # Verify we can checkout the old commit and get the original value
    assert commit1 is not None and commit2 is not None
    old_state = state.checkout(commit1)
    assert old_state is not None
    assert old_state.get("my_list") == [1, 2, 3]

    # Verify the new commit has the mutated value
    new_state = state.checkout(commit2)
    assert new_state is not None
    assert new_state.get("my_list") == [1, 2, 3, 4]


def test_mutation_detection_with_nested_objects():
    """Test mutation detection works with nested mutable objects."""
    store = kv.Memory()
    state = Versioned(store)

    # Set up nested mutable structure
    data = {"users": [{"name": "Alice", "scores": [10, 20]}], "config": {"debug": True}}
    state.set("app_data", data)
    commit1 = state.snapshot().commit_hash

    # Make nested mutations
    retrieved_data = state.get("app_data")
    retrieved_data["users"][0]["scores"].append(30)  # Deep mutation
    retrieved_data["config"]["timeout"] = 5000  # New key

    # Should detect and preserve mutations
    state.snapshot()

    # Verify mutations were preserved
    final_data = state.get("app_data")
    assert final_data["users"][0]["scores"] == [10, 20, 30]
    assert final_data["config"]["timeout"] == 5000

    # Verify old commit is unchanged
    assert commit1 is not None
    old_state = state.checkout(commit1)
    assert old_state is not None
    old_data = old_state.get("app_data")
    assert old_data["users"][0]["scores"] == [10, 20]
    assert "timeout" not in old_data["config"]


def test_meta_tracks_user_keys_and_sizes():
    import pickle

    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 123)
    state.set("__event_log__", ["ignore_me"])
    commit = state.snapshot().commit_hash

    meta_bytes = store.get(f"__meta__{commit}")
    total_bytes = store.get(f"__total_var_size__{commit}")
    assert meta_bytes is not None
    assert total_bytes is not None

    meta = pickle.loads(meta_bytes)
    total_size = pickle.loads(total_bytes)

    assert "a" in meta
    assert "__event_log__" not in meta

    last_touch = meta["a"].last_touch
    size = meta["a"].size
    assert last_touch > 0
    expected_size = len(pickle.dumps(123))
    assert size == expected_size
    assert total_size == expected_size


def test_meta_last_touch_persists_across_commits_and_reload():
    import pickle

    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    commit1 = state.snapshot().commit_hash
    meta1 = pickle.loads(store.get(f"__meta__{commit1}"))  # type: ignore[arg-type]
    touch1 = meta1["a"].last_touch
    size1 = meta1["a"].size
    assert touch1 > 0
    assert size1 == len(pickle.dumps(1))

    # Touch the key and force a new commit
    assert state.get("a") == 1
    state.set("b", 2)  # force a new snapshot
    commit2 = state.snapshot().commit_hash
    meta2 = pickle.loads(store.get(f"__meta__{commit2}"))  # type: ignore[arg-type]
    touch2 = meta2["a"].last_touch
    size2 = meta2["a"].size
    assert touch2 > touch1  # touch counter advanced
    assert size2 == size1

    # Reload to ensure metadata is hydrated
    Versioned(store)
    meta3_bytes = store.get(f"__meta__{commit2}")
    assert meta3_bytes is not None
    meta3 = pickle.loads(meta3_bytes)
    assert meta3["a"].last_touch == touch2
    assert meta3["b"].size == len(pickle.dumps(2))


def test_meta_removal_prunes_user_key():
    import pickle

    store = kv.Memory()
    state = Versioned(store)

    state.set("a", "keep?")
    state.snapshot()

    state.remove("a")
    commit = state.snapshot().commit_hash
    meta_bytes = store.get(f"__meta__{commit}")
    assert meta_bytes is not None
    meta = pickle.loads(meta_bytes)
    assert "a" not in meta


def test_mutation_updates_touch_and_size():
    import pickle

    store = kv.Memory()
    state = Versioned(store)

    data = [1, 2, 3]
    state.set("lst", data)
    commit1 = state.snapshot().commit_hash
    meta1 = pickle.loads(store.get(f"__meta__{commit1}"))  # type: ignore[arg-type]
    touch1 = meta1["lst"].last_touch
    size1 = meta1["lst"].size

    # Mutate in place
    retrieved = state.get("lst")
    retrieved.append(4)
    commit2 = state.snapshot().commit_hash
    meta2 = pickle.loads(store.get(f"__meta__{commit2}"))  # type: ignore[arg-type]
    touch2 = meta2["lst"].last_touch
    size2 = meta2["lst"].size

    assert touch2 > touch1  # mutation counts as a touch
    assert size2 > size1  # serialized size grows with added element


def test_base_commit_tracks_branch_start():
    """Test that base_commit is set correctly on init and updated after merge."""
    store = kv.Memory()
    state = Versioned(store)

    initial = state.current_commit
    assert state.base_commit == initial

    # Snapshot doesn't change base_commit
    state.set("a", 1)
    state.snapshot()
    assert state.base_commit == initial
    assert state.current_commit != initial

    # Merge updates base_commit to match current_commit
    state.merge()
    assert state.base_commit == state.current_commit


def test_snapshot_does_not_update_head():
    """Test that snapshot() does not update HEAD in the store."""
    import pickle

    from agex.state.versioned import HEAD_COMMIT

    store = kv.Memory()
    state = Versioned(store)
    initial = state.current_commit

    state.set("a", 1)
    state.snapshot()

    # HEAD should still be at initial
    head = pickle.loads(store.get(HEAD_COMMIT))
    assert head == initial
    assert state.current_commit != initial


def test_merge_updates_head():
    """Test that merge() updates HEAD in the store."""
    import pickle

    from agex.state.versioned import HEAD_COMMIT

    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    state.snapshot()
    state.merge()

    # HEAD should now match current_commit
    head = pickle.loads(store.get(HEAD_COMMIT))
    assert head == state.current_commit


def test_reset_reloads_from_head():
    """Test that reset() reloads state from current HEAD."""
    store = kv.Memory()
    state1 = Versioned(store)
    state2 = Versioned(store)

    # State1 makes changes and merges
    state1.set("a", 1)
    state1.snapshot()
    state1.merge()

    # State2 has local changes but calls reset
    state2.set("b", 2)
    state2.snapshot()
    state2.reset()

    # State2 should now see state1's data
    assert state2.get("a") == 1
    assert state2.get("b") is None
    assert state2.current_commit == state1.current_commit
    assert state2.base_commit == state1.current_commit


def test_revert_to_moves_head_to_earlier_commit():
    """Test that revert_to moves HEAD to an earlier commit."""
    import pickle

    from agex.state.versioned import HEAD_COMMIT

    store = kv.Memory()
    state = Versioned(store)

    # Create some history
    state.set("a", 1)
    state.snapshot()
    state.merge()
    commit1 = state.current_commit

    state.set("a", 2)
    state.snapshot()
    state.merge()

    state.set("a", 3)
    state.snapshot()
    state.merge()
    commit3 = state.current_commit

    # Verify we're at commit3
    assert state.get("a") == 3
    assert state.current_commit == commit3

    # Revert to commit1
    result = state.revert_to(commit1)
    assert result is True

    # Verify state is now at commit1
    assert state.get("a") == 1
    assert state.current_commit == commit1
    assert state.base_commit == commit1

    # HEAD should be updated in the store
    head = pickle.loads(store.get(HEAD_COMMIT))
    assert head == commit1


def test_revert_to_orphans_later_commits():
    """Test that revert_to leaves later commits as orphans."""
    store = kv.Memory()
    state = Versioned(store)

    # Create some history
    state.set("a", 1)
    state.snapshot()
    state.merge()
    commit1 = state.current_commit

    state.set("a", 2)
    state.snapshot()
    state.merge()
    commit2 = state.current_commit

    # Revert to commit1
    state.revert_to(commit1)

    # History should only go back to commit1, not to commit2
    history = list(state.history())
    assert commit1 in history
    assert commit2 not in history  # commit2 is now orphaned


def test_revert_to_returns_false_for_invalid_commit():
    """Test that revert_to returns False for commits not in history."""
    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    state.snapshot()
    state.merge()

    # Try to revert to a non-existent commit
    result = state.revert_to("nonexistent_hash")
    assert result is False

    # State should be unchanged
    assert state.get("a") == 1


def test_revert_to_clears_local_changes():
    """Test that revert_to clears any uncommitted local changes."""
    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    state.snapshot()
    state.merge()
    commit1 = state.current_commit

    # Make local changes without committing
    state.set("b", 2)

    # Revert to commit1
    state.revert_to(commit1)

    # Local changes should be gone
    assert state.get("a") == 1
    assert state.get("b") is None
    assert "b" not in list(state.keys())


def test_versioned_initial_commit():
    """Test that initial_commit returns the root commit hash."""
    store = kv.Memory()
    state = Versioned(store)

    # Initial state has one commit (the root)
    root = state.current_commit
    assert root is not None
    assert state.initial_commit == root

    # Add more commits
    state.set("a", 1)
    state.snapshot()
    commit1 = state.current_commit

    state.set("b", 2)
    state.snapshot()
    commit2 = state.current_commit

    # initial_commit should remain the same
    assert state.initial_commit == root
    assert state.initial_commit != commit1
    assert state.initial_commit != commit2

    # Revert to initial works
    state.revert_to(state.initial_commit)
    assert state.current_commit == root
    assert state.get("a") is None
    assert state.get("b") is None
