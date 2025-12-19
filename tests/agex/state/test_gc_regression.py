from agex.state.versioned import Versioned


def test_peek_vs_get_gc_behavior():
    """Verify that peek() does not update last_touch, while get() does."""
    store = Versioned()
    key = "user_fn"
    store.set(key, "function_body")

    # Get initial metadata
    meta_initial = store.meta[key]
    initial_touch = meta_initial.last_touch

    # Wait a bit? versioned.py uses an incrementing counter, so time.sleep isn't needed strictly
    # just next operation bumps it.

    # 1. PEEK should NOT update touch counter
    val = store.peek(key)
    assert val == "function_body"
    assert store.meta[key].last_touch == initial_touch

    # 2. GET SHOULD update touch counter
    val = store.get(key)
    assert val == "function_body"
    assert store.meta[key].last_touch > initial_touch


def test_namespaced_peek_delegation():
    """Verify peek works through Namespaced wrapper."""
    from agex.state.namespaced import Namespaced

    base = Versioned()
    ns = Namespaced(base, "ns")

    key = "foo"
    ns.set(key, "bar")

    # Verify internal key
    internal_key = "ns/foo"
    initial_touch = base.meta[internal_key].last_touch

    # Peek through namespace
    val = ns.peek(key)
    assert val == "bar"
    assert base.meta[internal_key].last_touch == initial_touch

    # Get through namespace
    val = ns.get(key)
    assert val == "bar"
    assert base.meta[internal_key].last_touch > initial_touch


def test_scoped_peek_delegation():
    """Verify peek works through Scoped wrapper."""
    from agex.state.scoped import Scoped

    base = Versioned()
    scoped = Scoped(base)

    key = "foo"
    base.set(key, "base_val")

    initial_touch = base.meta[key].last_touch

    # Peek through scoped (should delegate to base)
    val = scoped.peek(key)
    assert val == "base_val"
    assert base.meta[key].last_touch == initial_touch

    # Get through scoped (should delegate to base and update touch)
    val = scoped.get(key)
    assert val == "base_val"
    assert base.meta[key].last_touch > initial_touch


def test_closure_peek_delegation():
    """Verify peek works through LiveClosureState."""
    from agex.state.closure import LiveClosureState

    base = Versioned()
    key = "captured_var"
    base.set(key, "val")

    closure = LiveClosureState(base, {key})

    initial_touch = base.meta[key].last_touch

    # Peek through closure
    val = closure.peek(key)
    assert val == "val"
    assert base.meta[key].last_touch == initial_touch

    # Get through closure
    val = closure.get(key)
    assert val == "val"
    assert base.meta[key].last_touch > initial_touch
