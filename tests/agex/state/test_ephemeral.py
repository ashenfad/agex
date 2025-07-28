from agex.state import Live


def test_live_get_set_remove():
    state = Live()
    assert state.get("a") is None

    state.set("a", 1)
    assert state.get("a") == 1
    assert "a" in state

    state.remove("a")
    assert state.get("a") is None
    assert "a" not in state


def test_live_items():
    state = Live()
    state.set("a", 1)
    state.set("b", 2)

    assert dict(state.items()) == {"a": 1, "b": 2}
    assert list(state.keys()) == ["a", "b"]
    assert list(state.values()) == [1, 2]


def test_live_base_store():
    state = Live()
    assert state.base_store is state
