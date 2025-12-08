from agex.state import GCVersioned, kv


def test_rebase_noop_when_below_high_water():
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=10_000, low_water_bytes=8_000)
    state.set("a", "small")
    head = state.snapshot().commit_hash

    result = state.maybe_rebase()

    assert result.performed is False
    assert state.current_commit == head
    assert "a" in state.commit_keys


def test_rebase_drops_oldest_until_low_water():
    import os

    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=3_000, low_water_bytes=1_200)

    state.set("a", os.urandom(6000))  # oldest and largest
    state.set("b", os.urandom(2500))
    state.set("c", os.urandom(1500))
    state.snapshot()
    result = state.last_rebase_result

    assert result is not None
    assert result.performed is True
    assert result.total_size_after <= 1_200
    assert "a" in result.dropped_keys  # oldest, largest should go first
    assert "a" not in state.commit_keys


def test_rebase_retains_system_keys():
    import os

    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=3000, low_water_bytes=1200)
    state.set("__event_log__", ["keep"])
    state.set("payload", os.urandom(6000))
    state.snapshot()

    result = state.last_rebase_result

    assert "__event_log__" in state.commit_keys
    assert state.get("__event_log__") == ["keep"]
    assert result is not None
    assert "payload" in result.dropped_keys
    assert result.total_size_after <= 1200


def test_rebased_versioned_no_rebase_when_under_high_water():
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=5_000, low_water_bytes=4_000)
    state.set("x", "small")
    commit = state.snapshot().commit_hash

    assert state.current_commit == commit
    assert state.last_rebase_result is not None
    assert state.last_rebase_result.performed is False
    # Data remains
    assert state.get("x") == "small"


def test_rebased_versioned_rebases_after_snapshot():
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=1_200, low_water_bytes=800)

    state.set("a", "a" * 1000)
    state.set("b", "b" * 500)
    snap = state.snapshot()

    # Should have rebased (total size > high water)
    assert state.last_rebase_result is not None
    assert state.last_rebase_result.performed is True
    # Snapshot commit updated to the rebase head
    assert snap.commit_hash == state.current_commit

    # Coldest/largest should be dropped first
    assert "a" in state.last_rebase_result.dropped_keys
    assert state.get("a") is None
    # Still have some retained data
    assert state.get("b") == "b" * 500

    # Dropped blobs should be gone from the store
    # (we expect old versioned key for 'a' to be absent)
    old_commit = snap.commit_hash
    old_versioned_key = f"{old_commit}:a"
    assert store.get(old_versioned_key) is None


def test_rebase_drops_unreferenced_events():
    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=1_000, low_water_bytes=200)

    state.set("_event_keep", b"keep")
    state.set("_event_drop", b"drop")
    state.set("__event_log__", ["_event_keep"])
    state.snapshot()

    state.rebase()

    assert state.get("_event_keep") == b"keep"
    assert state.get("_event_drop") is None
    assert "_event_drop" not in state.commit_keys


def test_rebase_protects_referenced_events_from_gc():
    """Verify referenced events are never dropped even when exceeding high water."""
    import os

    store = kv.Memory()
    state = GCVersioned(store, high_water_bytes=3_000, low_water_bytes=1_000)

    # Create a large referenced event (oldest, largest - would be first GC candidate)
    large_event = os.urandom(5000)
    state.set("_event_old_large", large_event)
    state.set("__event_log__", ["_event_old_large"])
    state.snapshot()

    # Add newer user data that pushes us over high water
    state.set("user_data", os.urandom(2000))
    state.snapshot()

    # Rebase should have occurred (total > high water)
    result = state.last_rebase_result
    assert result is not None
    assert result.performed is True

    # Referenced event must NOT be dropped, even though it's oldest and largest
    assert "_event_old_large" not in result.dropped_keys
    assert "_event_old_large" in result.kept_keys
    assert state.get("_event_old_large") == large_event

    # User data should be dropped instead
    assert "user_data" in result.dropped_keys


def test_rebase_with_namespaced_state_protects_system_keys():
    """Verify system keys in namespaced states are protected from GC."""
    import os

    from agex.state import Namespaced

    store = kv.Memory()
    base_state = GCVersioned(store, high_water_bytes=3_000, low_water_bytes=1_000)

    # Create a namespaced state (simulating a sub-agent)
    ns_state = Namespaced(base_state, "sub_agent")

    # Add system keys and user data in the namespace
    ns_state.set("__event_log__", ["event1", "event2"])
    ns_state.set("__meta__", {"config": "value"})
    ns_state.set("large_data", os.urandom(5000))
    ns_state.set("small_data", "test")

    base_state.snapshot()

    result = base_state.last_rebase_result
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

    from agex.state import Namespaced

    store = kv.Memory()
    base_state = GCVersioned(store, high_water_bytes=2_000, low_water_bytes=800)

    # Create nested namespaces (parent → child → grandchild)
    parent = Namespaced(base_state, "parent")
    child = Namespaced(parent, "child")
    grandchild = Namespaced(child, "grandchild")

    # Add system key at deepest level
    grandchild.set("__event_log__", ["deep_event"])
    # Add large user data that will trigger GC
    grandchild.set("heavy_data", os.urandom(4000))
    parent.set("parent_data", "keep")

    base_state.snapshot()

    result = base_state.last_rebase_result
    assert result is not None
    assert result.performed is True

    # Deep system key should be protected (stored as "parent/child/grandchild/__event_log__")
    assert grandchild.get("__event_log__") == ["deep_event"]

    # Heavy user data should be dropped
    assert grandchild.get("heavy_data") is None


def test_rebase_preserves_event_log_across_multiple_namespaces():
    """Verify event logs in multiple namespaces are all protected."""
    import os

    from agex.state import Namespaced

    store = kv.Memory()
    base_state = GCVersioned(store, high_water_bytes=3_000, low_water_bytes=1_200)

    # Create multiple namespaced states (simulating multiple sub-agents)
    agent1 = Namespaced(base_state, "agent1")
    agent2 = Namespaced(base_state, "agent2")
    agent3 = Namespaced(base_state, "agent3")

    # Each agent has its own event log and user data
    agent1.set("__event_log__", ["agent1_event"])
    agent1.set("data", os.urandom(2000))

    agent2.set("__event_log__", ["agent2_event"])
    agent2.set("data", os.urandom(2000))

    agent3.set("__event_log__", ["agent3_event"])
    agent3.set("data", os.urandom(2000))

    base_state.snapshot()

    result = base_state.last_rebase_result
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

    from agex.state import Namespaced

    store = kv.Memory()
    base_state = GCVersioned(store, high_water_bytes=2_000, low_water_bytes=800)

    ns_state = Namespaced(base_state, "worker")

    # Mix of system keys and user variables with similar naming
    ns_state.set("__event_log__", ["system_event"])  # System key - should keep
    ns_state.set("__meta__", {"system": True})  # System key - should keep
    ns_state.set("event_data", os.urandom(3000))  # User var - can drop
    ns_state.set("meta_info", os.urandom(2000))  # User var - can drop

    base_state.snapshot()

    result = base_state.last_rebase_result
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
