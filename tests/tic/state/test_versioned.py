from tic.state import kv
from tic.state.versioned import Versioned


def test_versioned_get_set_remove():
    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    assert state.get("a") == 1
    state.remove("a")
    assert state.get("a") is None


def test_versioned_snapshot():
    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    h1 = state.snapshot()
    assert state.long_term.get(f"{h1}:a") == 1

    state.set("a", 2)
    h2 = state.snapshot()
    assert state.long_term.get(f"{h2}:a") == 2

    # Check that h1 is still there
    assert state.long_term.get(f"{h1}:a") == 1


def test_versioned_history():
    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    h1 = state.snapshot()
    state.set("b", 2)
    h2 = state.snapshot()

    history = list(state.history())
    assert history == [h2, h1]


def test_snapshot_creates_diff_keys():
    store = Versioned(kv.Memory())
    store.set("x", 1)
    store.set("y", 2)
    store.set("__internal__", "should be ignored")
    commit_hash = store.snapshot()

    commit_state = store.checkout(commit_hash)
    diff_keys = commit_state.get("__diff_keys__")
    assert diff_keys == ("x", "y")


def test_diffs():
    store = Versioned(kv.Memory())
    store.set("a", 100)
    store.snapshot()

    # First set of changes
    store.set("x", 1)
    store.set("y", 2)
    store.set("__stdout__", ["hello"])
    commit1 = store.snapshot()

    # Second set of changes
    store.set("y", 3)
    store.set("z", 4)
    store.set("__stdout__", ["world"])
    commit2 = store.snapshot()

    # Check changes for commit 1
    state_changes = store.diffs(commit1)
    assert state_changes == {"x": 1, "y": 2}
    assert store.checkout(commit1).get("__stdout__") == ["hello"]

    # Check changes for commit 2
    state_changes_2 = store.diffs(commit2)
    assert state_changes_2 == {"y": 3, "z": 4}

    # Check changes for the most recent commit (default)
    state_changes_3 = store.diffs()
    assert state_changes_3 == state_changes_2
    assert store.get("__stdout__") == ["world"]


def test_snapshot_on_empty_ephemeral_does_not_create_commit():
    store = Versioned(kv.Memory())
    commit1 = store.snapshot()
    assert commit1 is None

    store.set("a", 1)
    commit2 = store.snapshot()
    assert commit2 is not None

    commit3 = store.snapshot()
    assert commit2 == commit3
