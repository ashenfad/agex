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
