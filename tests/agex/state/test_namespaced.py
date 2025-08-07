import pytest

from agex.state import Namespaced, Versioned, kv


def test_namespaced_get_set():
    store = kv.Memory()
    state = Versioned(store)
    ns1 = Namespaced(state, "ns1")

    ns1.set("a", 1)
    assert ns1.get("a") == 1


def test_namespaced_isolation():
    state = Versioned(kv.Memory())
    ns1 = Namespaced(state, "ns1")
    ns2 = Namespaced(state, "ns2")

    ns1.set("a", 1)
    ns2.set("a", 2)

    assert ns1.get("a") == 1
    assert ns2.get("a") == 2


def test_nested_namespaces():
    state = Versioned(kv.Memory())
    top = Namespaced(state, "top")
    bottom = Namespaced(top, "bottom")

    top.set("a", 1)
    bottom.set("a", 2)

    assert top.get("a") == 1
    assert bottom.get("a") == 2

    state.snapshot()

    # Recreate from scratch to test reading
    top2 = Namespaced(state, "top")
    bottom2 = Namespaced(top2, "bottom")

    assert top2.get("a") == 1
    assert bottom2.get("a") == 2


def test_namespaced_base_store():
    state = Versioned(kv.Memory())
    ns1 = Namespaced(state, "ns1")
    ns2 = Namespaced(ns1, "ns2")

    assert ns2.base_store is state


def test_namespace_disallows_slash():
    state = Versioned(kv.Memory())
    with pytest.raises(ValueError):
        Namespaced(state, "a/b")


def test_full_path_tracking():
    """Test that namespace path is properly tracked for nested namespaces."""
    state = Versioned(kv.Memory())

    # Single level
    ns1 = Namespaced(state, "agent")
    assert ns1.namespace == "agent"

    # Nested levels
    ns2 = Namespaced(ns1, "worker")
    assert ns2.namespace == "agent/worker"

    ns3 = Namespaced(ns2, "task")
    assert ns3.namespace == "agent/worker/task"


def test_namespace_collision_prevention():
    """Test that similar namespace paths don't collide."""
    state = Versioned(kv.Memory())

    # Create overlapping namespace hierarchies:
    # Path 1: a/b
    # Path 2: a/c/b
    ns_a = Namespaced(state, "a")
    ns_ab = Namespaced(ns_a, "b")

    ns_ac = Namespaced(ns_a, "c")
    ns_acb = Namespaced(ns_ac, "b")

    # Set values in both "b" namespaces
    ns_ab.set("data", "from_a_b")
    ns_acb.set("data", "from_a_c_b")

    # Verify isolation - each should only see its own data
    assert ns_ab.get("data") == "from_a_b"
    assert ns_acb.get("data") == "from_a_c_b"

    # Verify keys() shows only local data
    ab_keys = list(ns_ab.keys())
    acb_keys = list(ns_acb.keys())

    assert ab_keys == ["data"]
    assert acb_keys == ["data"]

    # Verify no cross-contamination
    assert "data" in ns_ab
    assert "data" in ns_acb
    assert ns_ab.get("nonexistent") is None
    assert ns_acb.get("nonexistent") is None


def test_complex_namespace_isolation():
    """Test isolation with multiple overlapping paths."""
    state = Versioned(kv.Memory())

    # Create multiple agents with similar path structures
    orchestrator = Namespaced(state, "orchestrator")
    worker1 = Namespaced(orchestrator, "worker")

    # Different orchestrator lineage
    other_orch = Namespaced(state, "other")
    worker2 = Namespaced(other_orch, "worker")

    # Same immediate namespace name but different full paths
    worker1.set("task", "task1")
    worker1.set("status", "running")

    worker2.set("task", "task2")
    worker2.set("status", "idle")

    # Verify complete isolation
    assert worker1.get("task") == "task1"
    assert worker1.get("status") == "running"

    assert worker2.get("task") == "task2"
    assert worker2.get("status") == "idle"

    # Verify keys are properly isolated
    w1_keys = set(worker1.keys())
    w2_keys = set(worker2.keys())

    assert w1_keys == {"task", "status"}
    assert w2_keys == {"task", "status"}

    # Verify no cross-namespace access
    assert worker1.get("nonexistent") is None
    assert worker2.get("nonexistent") is None


def test_local_namespace_detection():
    """Test the _local_namespace method works correctly with full paths."""
    state = Versioned(kv.Memory())

    # Create nested namespace
    ns_a = Namespaced(state, "a")
    ns_ab = Namespaced(ns_a, "b")

    # Test keys that should and shouldn't match
    test_cases = [
        ("a/b/data", "data"),  # Direct child - should be visible
        ("a/b/config", "config"),  # Another direct child - should be visible
        ("a/b/sub/task", None),  # Child namespace - should be hidden
        ("a/b/worker/status", None),  # Another child namespace - should be hidden
        ("a/b/", None),  # Empty remainder
        ("a/c/data", None),  # Different sibling path
        ("a/c/b/data", None),  # Similar but longer path
        ("other/data", None),  # Completely different path
        ("a/b", None),  # Path without trailing slash/content
    ]

    for key, expected in test_cases:
        result = ns_ab._local_namespace(key)
        assert result == expected, (
            f"Key '{key}' should return '{expected}', got '{result}'"
        )


def test_keys_only_shows_direct_children():
    """Test that keys() only shows direct keys, not sub-namespace keys (filesystem-like behavior)."""
    state = Versioned(kv.Memory())

    # Create namespace
    ns_ab = Namespaced(Namespaced(state, "a"), "b")

    # Add direct keys to a/b
    ns_ab.set("config", "value1")
    ns_ab.set("data", "value2")
    ns_ab.set("status", "active")

    # Add keys in sub-namespaces (should be hidden from a/b keys())
    state.set("a/b/worker/task", "work1")  # Sub-namespace: a/b/worker/
    state.set("a/b/worker/status", "running")  # Sub-namespace: a/b/worker/
    state.set("a/b/cache/item1", "cached")  # Sub-namespace: a/b/cache/
    state.set("a/b/logs/error.log", "errors")  # Sub-namespace: a/b/logs/

    # Get keys from a/b namespace
    ab_keys = set(ns_ab.keys())

    # Should only see direct keys, not sub-namespace keys
    expected_keys = {"config", "data", "status"}
    assert ab_keys == expected_keys, f"Expected {expected_keys}, got {ab_keys}"

    # Verify we can still access direct keys
    assert ns_ab.get("config") == "value1"
    assert ns_ab.get("data") == "value2"
    assert ns_ab.get("status") == "active"

    # Verify sub-namespace keys are not accessible through this namespace
    assert ns_ab.get("worker") is None  # worker is a sub-namespace, not a key
    assert ns_ab.get("cache") is None  # cache is a sub-namespace, not a key


def test_mixed_direct_and_nested_keys():
    """Test complex scenario with both direct keys and nested namespaces."""
    state = Versioned(kv.Memory())

    # Create base namespace
    ns_root = Namespaced(state, "root")

    # Add direct keys to root
    ns_root.set("app_config", "config_value")
    ns_root.set("version", "1.0.0")

    # Create sub-namespaces and add keys
    ns_agents = Namespaced(ns_root, "agents")
    ns_agents.set("count", 3)
    ns_agents.set("active", True)

    ns_worker1 = Namespaced(ns_agents, "worker1")
    ns_worker1.set("task", "processing")
    ns_worker1.set("status", "busy")

    ns_worker2 = Namespaced(ns_agents, "worker2")
    ns_worker2.set("task", "idle")

    # Test root namespace only sees its direct keys
    root_keys = set(ns_root.keys())
    assert root_keys == {
        "app_config",
        "version",
    }, f"Root keys should be direct only, got {root_keys}"

    # Test agents namespace only sees its direct keys
    agents_keys = set(ns_agents.keys())
    assert agents_keys == {
        "count",
        "active",
    }, f"Agents keys should be direct only, got {agents_keys}"

    # Test worker namespaces see their own keys
    worker1_keys = set(ns_worker1.keys())
    assert worker1_keys == {
        "task",
        "status",
    }, f"Worker1 keys should be direct only, got {worker1_keys}"

    worker2_keys = set(ns_worker2.keys())
    assert worker2_keys == {"task"}, (
        f"Worker2 keys should be direct only, got {worker2_keys}"
    )

    # Verify isolation - each namespace only sees its own direct keys
    assert ns_root.get("app_config") == "config_value"
    assert ns_root.get("count") is None  # This is in agents sub-namespace

    assert ns_agents.get("count") == 3
    assert ns_agents.get("app_config") is None  # This is in parent namespace
    assert ns_agents.get("task") is None  # This is in worker sub-namespaces


def test_descendant_keys_hierarchical_traversal():
    """Test descendant_keys() returns all keys including nested namespaces."""
    state = Versioned(kv.Memory())

    # Create namespace
    ns_ab = Namespaced(Namespaced(state, "a"), "b")

    # Add direct keys
    ns_ab.set("config", "value1")
    ns_ab.set("data", "value2")

    # Add nested namespace keys (via base store to simulate sub-agents)
    state.set("a/b/worker/task", "work1")
    state.set("a/b/worker/status", "running")
    state.set("a/b/cache/item1", "cached")
    state.set("a/b/logs/error.log", "errors")
    state.set("a/b/logs/access.log", "requests")

    # Test keys() vs descendant_keys()
    direct_keys = set(ns_ab.keys())
    all_keys = set(ns_ab.descendant_keys())

    # keys() should only show direct keys
    assert direct_keys == {"config", "data"}

    # descendant_keys() should show all keys including nested paths
    expected_all_keys = {
        "config",
        "data",  # Direct keys
        "worker/task",
        "worker/status",  # worker sub-namespace
        "cache/item1",  # cache sub-namespace
        "logs/error.log",
        "logs/access.log",  # logs sub-namespace
    }
    assert all_keys == expected_all_keys, (
        f"Expected {expected_all_keys}, got {all_keys}"
    )


def test_descendant_keys_nested_namespace_isolation():
    """Test descendant_keys() respects namespace boundaries."""
    state = Versioned(kv.Memory())

    # Create multiple top-level namespaces
    ns_a = Namespaced(state, "a")
    ns_b = Namespaced(state, "b")

    # Create nested namespaces under 'a'
    ns_a_worker = Namespaced(ns_a, "worker")
    ns_a_cache = Namespaced(ns_a, "cache")

    # Add keys to 'a' namespace and sub-namespaces
    ns_a.set("config", "a_config")
    ns_a_worker.set("task", "a_work")
    ns_a_worker.set("status", "busy")
    ns_a_cache.set("item", "cached_data")

    # Add keys to 'b' namespace (should be isolated)
    ns_b.set("config", "b_config")
    ns_b.set("data", "b_data")

    # Test descendant_keys() for 'a' namespace
    a_descendants = set(ns_a.descendant_keys())
    expected_a_descendants = {
        "config",  # Direct key in 'a'
        "worker/task",  # Key in 'a/worker'
        "worker/status",  # Key in 'a/worker'
        "cache/item",  # Key in 'a/cache'
    }
    assert a_descendants == expected_a_descendants, (
        f"Expected {expected_a_descendants}, got {a_descendants}"
    )

    # Test descendant_keys() for 'b' namespace
    b_descendants = set(ns_b.descendant_keys())
    expected_b_descendants = {"config", "data"}  # Only direct keys in 'b'
    assert b_descendants == expected_b_descendants, (
        f"Expected {expected_b_descendants}, got {b_descendants}"
    )

    # Test descendant_keys() for nested namespace
    worker_descendants = set(ns_a_worker.descendant_keys())
    expected_worker_descendants = {"task", "status"}  # Only keys in 'a/worker'
    assert worker_descendants == expected_worker_descendants, (
        f"Expected {expected_worker_descendants}, got {worker_descendants}"
    )


def test_descendant_keys_empty_namespace():
    """Test descendant_keys() on namespace with no keys."""
    state = Versioned(kv.Memory())
    ns_empty = Namespaced(state, "empty")

    # Should return empty iterable
    descendants = list(ns_empty.descendant_keys())
    assert descendants == [], (
        f"Empty namespace should have no descendants, got {descendants}"
    )

    # Add some unrelated keys
    state.set("other/key", "value")
    state.set("different/namespace/key", "value")

    # Should still return empty
    descendants = list(ns_empty.descendant_keys())
    assert descendants == [], f"Unrelated keys should not appear, got {descendants}"


def test_descendant_keys_vs_keys_comparison():
    """Test the difference between keys() and descendant_keys() in various scenarios."""
    state = Versioned(kv.Memory())

    # Test 1: Only direct keys
    ns1 = Namespaced(state, "test1")
    ns1.set("direct1", "value1")
    ns1.set("direct2", "value2")

    assert set(ns1.keys()) == {"direct1", "direct2"}
    assert set(ns1.descendant_keys()) == {"direct1", "direct2"}

    # Test 2: Only nested keys
    ns2 = Namespaced(state, "test2")
    state.set("test2/sub/nested1", "value1")
    state.set("test2/sub/nested2", "value2")

    assert set(ns2.keys()) == set()  # No direct keys
    assert set(ns2.descendant_keys()) == {"sub/nested1", "sub/nested2"}

    # Test 3: Mixed direct and nested keys
    ns3 = Namespaced(state, "test3")
    ns3.set("direct", "direct_value")
    state.set("test3/nested/key", "nested_value")

    assert set(ns3.keys()) == {"direct"}
    assert set(ns3.descendant_keys()) == {"direct", "nested/key"}
