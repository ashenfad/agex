import pytest

from tic.render.view import view
from tic.state import kv
from tic.state.versioned import Versioned


def test_view_full():
    store = Versioned(kv.Memory())
    store.set("x", 1)
    store.snapshot()
    store.set("y", "hello")
    store.snapshot()

    full_state = view(store, focus="full")
    assert full_state == {"x": 1, "y": "hello"}


def test_view_stdout():
    store = Versioned(kv.Memory())
    store.set("__stdout__", ["a"])
    store.snapshot()
    store.set("__stdout__", ["b"])
    store.snapshot()

    stdout = view(store, focus="stdout")
    assert stdout == ["b"]


def test_view_recent_shows_last_commit_state_only():
    store = Versioned(kv.Memory())
    store.set("x", 100)
    store.snapshot()

    store.set("y", 200)
    store.set("__stdout__", ["change"])
    store.set("__internal__", "value")
    store.snapshot()

    recent_view = view(store, focus="recent")
    assert "y = 200" in recent_view
    assert "change" not in recent_view
    assert "__internal__" not in recent_view
    assert "Agent printed" not in recent_view
    assert "x = 100" not in recent_view


def test_view_raises_on_hot_storage():
    store = Versioned(kv.Memory())
    store.set("x", 1)

    with pytest.raises(ValueError, match="uncommitted ephemeral changes"):
        view(store, focus="recent")


def test_view_recent_with_token_budgets():
    store = Versioned(kv.Memory())
    store.set("y", "b" * 1000)
    store.snapshot()

    recent_view_small = view(store, focus="recent", max_tokens=30)
    assert len(recent_view_small) < 200
    assert "y =" in recent_view_small
    assert "..." in recent_view_small
    assert "b" * 50 not in recent_view_small
