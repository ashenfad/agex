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
