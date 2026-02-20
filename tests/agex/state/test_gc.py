from kvit import Namespaced, Staged
from kvit.gc import GCVersioned

from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def _make_gc_state(store=None, **gc_kwargs):
    """Create a Staged store wrapping a GCVersioned with agex codecs."""
    if store is None:
        store = Memory()
    versioned = GCVersioned(store, **gc_kwargs)
    return Staged(versioned, encoder=_agex_encoder, decoder=_agex_decoder)


def test_rebase_noop_when_below_high_water():
    store = Memory()
    state = _make_gc_state(store, high_water_bytes=10_000, low_water_bytes=8_000)
    state.set("a", "small")
    state.commit()

    gc_result = state.versioned.maybe_rebase()

    assert gc_result.performed is False
    assert "a" in state


def test_rebase_drops_oldest_until_low_water():
    import os

    store = Memory()
    state = _make_gc_state(store, high_water_bytes=3_000, low_water_bytes=1_200)

    state.set("a", os.urandom(6000))  # oldest and largest
    state.set("b", os.urandom(2500))
    state.set("c", os.urandom(1500))
    state.commit()
    result = state.versioned.last_rebase_result

    assert result is not None
    assert result.performed is True
    assert result.total_size_after <= 1_200
    assert "a" in result.dropped_keys  # oldest, largest should go first
    assert "a" not in state


def test_rebase_retains_system_keys():
    import os

    store = Memory()
    state = _make_gc_state(store, high_water_bytes=3000, low_water_bytes=1200)
    state.set("__event_log__", ["keep"])
    state.set("payload", os.urandom(6000))
    state.commit()

    result = state.versioned.last_rebase_result

    assert "__event_log__" in state
    assert state.get("__event_log__") == ["keep"]
    assert result is not None
    assert "payload" in result.dropped_keys
    assert result.total_size_after <= 1200


def test_rebased_versioned_no_rebase_when_under_high_water():
    store = Memory()
    state = _make_gc_state(store, high_water_bytes=5_000, low_water_bytes=4_000)
    state.set("x", "small")
    state.commit()

    assert state.versioned.last_rebase_result is not None
    assert state.versioned.last_rebase_result.performed is False
    # Data remains
    assert state.get("x") == "small"


def test_rebased_versioned_rebases_after_snapshot():
    store = Memory()
    state = _make_gc_state(store, high_water_bytes=1_200, low_water_bytes=800)

    state.set("a", "a" * 1000)
    state.set("b", "b" * 500)
    state.commit()

    # Should have rebased (total size > high water)
    assert state.versioned.last_rebase_result is not None
    assert state.versioned.last_rebase_result.performed is True

    # Coldest/largest should be dropped first
    assert "a" in state.versioned.last_rebase_result.dropped_keys
    assert state.get("a") is None
    # Still have some retained data
    assert state.get("b") == "b" * 500


def test_rebase_with_namespaced_state_protects_system_keys():
    """Verify system keys in namespaced states are protected from GC."""
    import os

    store = Memory()
    state = _make_gc_state(store, high_water_bytes=3_000, low_water_bytes=1_000)

    # Create a namespaced state (simulating a sub-agent)
    ns_state = Namespaced(state, "sub_agent")

    # Add system keys and user data in the namespace
    ns_state.set("__event_log__", ["event1", "event2"])
    ns_state.set("__meta__", {"config": "value"})
    ns_state.set("large_data", os.urandom(5000))
    ns_state.set("small_data", "test")

    state.commit()

    result = state.versioned.last_rebase_result
    assert result is not None
    assert result.performed is True

    # System keys should be protected (stored as "sub_agent/__event_log__" etc.)
    assert ns_state.get("__event_log__") == ["event1", "event2"]
    assert ns_state.get("__meta__") == {"config": "value"}

    # User data should be dropped (keys include namespace prefix)
    assert "sub_agent/large_data" in result.dropped_keys
    assert ns_state.get("large_data") is None


def test_rebase_with_nested_namespaces():
    """Verify deeply nested namespaces work correctly with GC."""
    import os

    store = Memory()
    state = _make_gc_state(store, high_water_bytes=2_000, low_water_bytes=800)

    # Create nested namespaces (parent -> child -> grandchild)
    parent = Namespaced(state, "parent")
    child = Namespaced(parent, "child")
    grandchild = Namespaced(child, "grandchild")

    # Add system key at deepest level
    grandchild.set("__event_log__", ["deep_event"])
    # Add large user data that will trigger GC
    grandchild.set("heavy_data", os.urandom(4000))
    parent.set("parent_data", "keep")

    state.commit()

    result = state.versioned.last_rebase_result
    assert result is not None
    assert result.performed is True

    # Deep system key should be protected (stored as "parent/child/grandchild/__event_log__")
    assert grandchild.get("__event_log__") == ["deep_event"]

    # Heavy user data should be dropped
    assert grandchild.get("heavy_data") is None


def test_rebase_preserves_event_log_across_multiple_namespaces():
    """Verify event logs in multiple namespaces are all protected."""
    import os

    store = Memory()
    state = _make_gc_state(store, high_water_bytes=3_000, low_water_bytes=1_200)

    # Create multiple namespaced states (simulating multiple sub-agents)
    agent1 = Namespaced(state, "agent1")
    agent2 = Namespaced(state, "agent2")
    agent3 = Namespaced(state, "agent3")

    # Each agent has its own event log and user data
    agent1.set("__event_log__", ["agent1_event"])
    agent1.set("data", os.urandom(2000))

    agent2.set("__event_log__", ["agent2_event"])
    agent2.set("data", os.urandom(2000))

    agent3.set("__event_log__", ["agent3_event"])
    agent3.set("data", os.urandom(2000))

    state.commit()

    result = state.versioned.last_rebase_result
    assert result is not None
    assert result.performed is True

    # All event logs should be protected
    assert agent1.get("__event_log__") == ["agent1_event"]
    assert agent2.get("__event_log__") == ["agent2_event"]
    assert agent3.get("__event_log__") == ["agent3_event"]

    # At least some user data should be dropped
    dropped_count = sum(
        [
            agent1.get("data") is None,
            agent2.get("data") is None,
            agent3.get("data") is None,
        ]
    )
    assert dropped_count >= 2  # Should drop at least 2 out of 3 to get under low_water


def test_rebase_drops_namespaced_user_vars_not_system_keys():
    """Verify GC correctly distinguishes between user vars and system keys in namespaces."""
    import os

    store = Memory()
    state = _make_gc_state(store, high_water_bytes=2_000, low_water_bytes=800)

    ns_state = Namespaced(state, "worker")

    # Mix of system keys and user variables with similar naming
    ns_state.set("__event_log__", ["system_event"])  # System key - should keep
    ns_state.set("__meta__", {"system": True})  # System key - should keep
    ns_state.set("event_data", os.urandom(3000))  # User var - can drop
    ns_state.set("meta_info", os.urandom(2000))  # User var - can drop

    state.commit()

    result = state.versioned.last_rebase_result
    assert result is not None
    assert result.performed is True

    # System keys preserved
    assert ns_state.get("__event_log__") == ["system_event"]
    assert ns_state.get("__meta__") == {"system": True}

    # User variables should be dropped
    assert ns_state.get("event_data") is None
    assert ns_state.get("meta_info") is None

    # Verify the dropped keys were actually user vars (with namespace prefix)
    assert "worker/event_data" in result.dropped_keys
    assert "worker/meta_info" in result.dropped_keys
