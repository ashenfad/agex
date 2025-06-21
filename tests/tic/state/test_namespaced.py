import pytest

from tic.state import Namespaced, Versioned, kv


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
