from tic.state import Versioned, kv


def test_versioned_get_set_snapshot():
    store = kv.Memory()
    state = Versioned(store)

    state.set("a", 1)
    assert state.get("a") == 1

    # Value is not in the store until snapshotted
    assert store.get(state._versioned_key("a", state.current_commit)) is None

    h1 = state.snapshot()
    assert state.get("a") == 1
    assert store.get(state._versioned_key("a", h1)) == 1


def test_versioned_forking():
    store = kv.Memory()
    state = Versioned(store)
    state.set("a", 1)
    h1 = state.snapshot()

    # Create a new state from the old hash
    state2 = Versioned(store, commit_hash=h1)
    assert state2.get("a") == 1

    # Modify the new state
    state2.set("a", 2)
    assert state2.get("a") == 2

    # The original state is unaffected
    assert state.get("a") == 1


def test_versioned_history():
    store = kv.Memory()
    state = Versioned(store)

    h1 = state.snapshot()
    state.set("a", 1)
    h2 = state.snapshot()
    state.set("b", 2)
    h3 = state.snapshot()

    history = list(state.history())
    assert history == [h3, h2, h1]
