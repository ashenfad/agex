import pytest

from tic.agent import Agent, MemberSpec
from tic.render.view import view
from tic.state import kv
from tic.state.versioned import Versioned


def test_view_agent_default():
    agent = Agent()

    class MyClass:
        def high_vis_method(self):
            """High visibility"""
            pass

        def medium_vis_method(self):
            pass

        def low_vis_method(self):
            pass

    agent.cls(
        MyClass,
        include=["high_vis_method", "medium_vis_method", "low_vis_method"],
        configure={
            "high_vis_method": MemberSpec(visibility="high"),
            "medium_vis_method": MemberSpec(visibility="medium"),
            "low_vis_method": MemberSpec(visibility="low"),
        },
    )

    output = view(agent)
    assert "high_vis_method" in output
    assert "medium_vis_method" in output
    assert "low_vis_method" not in output
    # Medium-vis methods should not show their docstrings
    assert "High visibility" in output
    assert "..." in output


def test_view_agent_full():
    agent = Agent()

    class MyClass:
        def high_vis_method(self):
            """High visibility"""
            pass

        def medium_vis_method(self):
            pass

        def low_vis_method(self):
            """Low visibility"""
            pass

    agent.cls(
        MyClass,
        visibility="low",  # Set class low to test promotion
        include=["high_vis_method", "medium_vis_method", "low_vis_method"],
        configure={
            "high_vis_method": MemberSpec(visibility="high"),
            "medium_vis_method": MemberSpec(visibility="medium"),
            "low_vis_method": MemberSpec(visibility="low"),
        },
    )

    output = view(agent, full=True)
    # Full view should render everything, including docstrings
    assert "high_vis_method" in output
    assert "medium_vis_method" in output
    assert "low_vis_method" in output
    assert "High visibility" in output
    assert "Low visibility" in output
    assert "..." not in output


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
