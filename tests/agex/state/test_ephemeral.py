from agex.state import Ephemeral


def test_ephemeral_get_set_remove():
    state = Ephemeral()
    assert state.get("a") is None

    state.set("a", 1)
    assert state.get("a") == 1
    assert "a" in state

    state.remove("a")
    assert state.get("a") is None
    assert "a" not in state


def test_ephemeral_items():
    state = Ephemeral()
    state.set("a", 1)
    state.set("b", 2)

    assert dict(state.items()) == {"a": 1, "b": 2}
    assert list(state.keys()) == ["a", "b"]
    assert list(state.values()) == [1, 2]


def test_ephemeral_base_store():
    state = Ephemeral()
    assert state.base_store is state
