from agex.state import get_root
from agex.state.live import Live


def test_live_get_set_remove():
    state = Live()
    assert state.get("a") is None

    state["a"] = 1
    assert state.get("a") == 1
    assert "a" in state

    del state["a"]
    assert state.get("a") is None
    assert "a" not in state


def test_live_items():
    state = Live()
    state["a"] = 1
    state["b"] = 2

    assert dict(state.items()) == {"a": 1, "b": 2}
    assert set(state.keys()) == {"a", "b"}
    assert set(state.values()) == {1, 2}


def test_live_root_is_self():
    state = Live()
    assert get_root(state) is state
