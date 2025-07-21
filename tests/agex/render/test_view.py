import pytest

from agex.agent import Agent, MemberSpec
from agex.render.view import view
from agex.state import kv
from agex.state.versioned import Versioned


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
    # Full view should render everything, including docstrings for high/low vis
    # and ... for medium vis.
    assert "high_vis_method" in output
    assert "medium_vis_method" in output
    assert "low_vis_method" in output
    assert "High visibility" in output
    assert "Low visibility" in output
    assert "..." in output


def test_view_full():
    store = Versioned(kv.Memory())
    store.set("x", 1)
    store.snapshot()
    store.set("y", "hello")
    store.snapshot()

    full_state = view(store, focus="full")
    assert full_state == {"x": 1, "y": "hello"}


def test_view_events():
    from agex.agent.events import OutputEvent
    from agex.state.log import add_event_to_log

    store = Versioned(kv.Memory())

    # Add first event and snapshot
    event_a = OutputEvent(agent_name="test_agent", parts=["event_a"])
    add_event_to_log(store, event_a)
    store.snapshot()

    # Add second event and snapshot
    event_b = OutputEvent(agent_name="test_agent", parts=["event_b"])
    add_event_to_log(store, event_b)
    store.snapshot()

    events = view(store, focus="events")
    assert isinstance(events, list)
    assert len(events) == 2
    assert isinstance(events[0], OutputEvent)
    assert isinstance(events[1], OutputEvent)
    assert events[0].parts == ["event_a"]
    assert events[1].parts == ["event_b"]


def test_view_recent_shows_last_commit_state_only():
    store = Versioned(kv.Memory())
    store.set("x", 100)
    store.snapshot()

    store.set("y", 200)
    store.snapshot()

    recent_view = view(store, focus="recent")
    assert "y = 200" in recent_view
    assert "__internal__" not in recent_view
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
