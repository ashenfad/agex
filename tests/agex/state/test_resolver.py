"""Tests for bring-your-own state resolvers (connect_state type='resolver')."""

import pytest
from kvgit.kv import Memory

from agex.host.local import Local
from agex.state import (
    SAFE_SESSION_RE,
    StateResolver,
    assert_safe_session,
    connect_state,
    staged_state,
)


class BranchResolver:
    """One shared store, a kvgit branch per session."""

    versioned = True

    def __init__(self, store):
        self._store = store
        self._cache = {}

    def resolve(self, session):
        assert_safe_session(session)
        if session not in self._cache:
            self._cache[session] = staged_state(self._store, branch=session)
        return self._cache[session]


# ---------------------------------------------------------------------------
# connect_state validation
# ---------------------------------------------------------------------------


def test_connect_state_resolver_roundtrip():
    resolver = BranchResolver(Memory())
    config = connect_state(type="resolver", resolver=resolver)
    assert config.type == "resolver"
    assert config.resolver is resolver
    assert config.storage is None


def test_connect_state_resolver_requires_resolver():
    with pytest.raises(ValueError, match="requires a 'resolver'"):
        connect_state(type="resolver")


def test_connect_state_resolver_rejects_non_resolver():
    with pytest.raises(ValueError, match="resolve\\(session\\)"):
        connect_state(type="resolver", resolver=object())


def test_connect_state_resolver_rejects_storage():
    resolver = BranchResolver(Memory())
    with pytest.raises(ValueError, match="storage"):
        connect_state(type="resolver", storage="memory", resolver=resolver)


def test_connect_state_resolver_rejects_init():
    resolver = BranchResolver(Memory())
    with pytest.raises(ValueError, match="init"):
        connect_state(type="resolver", resolver=resolver, init={"x": 1})


def test_resolver_config_refuses_remote_serialization():
    config = connect_state(type="resolver", resolver=BranchResolver(Memory()))
    with pytest.raises(ValueError, match="Local-host only"):
        config.dump_config()


# ---------------------------------------------------------------------------
# Local host delegation
# ---------------------------------------------------------------------------


def test_local_host_delegates_to_resolver():
    resolver = BranchResolver(Memory())
    config = connect_state(type="resolver", resolver=resolver)
    host = Local()
    host.validate_state(config)

    state = host.resolve_state(config, "chat-1")
    assert state is resolver.resolve("chat-1")  # resolver owns the cache

    # No host-side caching on the resolver path: repeated host calls go
    # back through the resolver every time.
    assert host.resolve_state(config, "chat-1") is state
    assert not host._session_cache


def test_local_host_resolver_sessions_are_isolated():
    resolver = BranchResolver(Memory())
    config = connect_state(type="resolver", resolver=resolver)
    host = Local()

    s1 = host.resolve_state(config, "chat-1")
    s2 = host.resolve_state(config, "chat-2")
    s1["x"] = 1
    s1.commit()
    assert s2.get("x") is None  # separate branches over one substrate


def test_shared_branch_concurrent_working_trees_merge():
    """Two working trees on one branch: disjoint commits auto-merge."""
    store = Memory()
    t1 = staged_state(store, branch="shared")
    t2 = staged_state(store, branch="shared")

    t1["a"] = 1
    t1.commit()
    t2["b"] = 2
    t2.commit()  # CAS sees moved HEAD, three-way merges

    fresh = staged_state(store, branch="shared")
    assert fresh.get("a") == 1
    assert fresh.get("b") == 2


def test_agent_state_uses_resolver():
    from agex import Agent, clear_agent_registry

    clear_agent_registry()
    resolver = BranchResolver(Memory())
    agent = Agent(
        name="resolver_test",
        state=connect_state(type="resolver", resolver=resolver),
    )
    assert agent.state("chat-1") is resolver.resolve("chat-1")


# ---------------------------------------------------------------------------
# Remote hosts reject resolver configs
# ---------------------------------------------------------------------------


def test_http_host_rejects_resolver():
    from agex.host.http import HTTP

    config = connect_state(type="resolver", resolver=BranchResolver(Memory()))
    with pytest.raises(ValueError, match="not supported on the HTTP host"):
        HTTP("http://localhost:8000").validate_state(config)


def test_modal_validate_rejects_resolver():
    from agex.host.modal import _validate_modal_state

    config = connect_state(type="resolver", resolver=BranchResolver(Memory()))
    with pytest.raises(ValueError, match="not supported on Modal"):
        _validate_modal_state(config)


# ---------------------------------------------------------------------------
# Session-id validation (path traversal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", "../evil", "..", ".hidden", "a/b", "a\\b", "a b", "a\x00b", ".", "/abs"],
)
def test_assert_safe_session_rejects(bad):
    with pytest.raises(ValueError, match="invalid session id"):
        assert_safe_session(bad)


@pytest.mark.parametrize("good", ["default", "chat-1", "user_alice", "a.b-c_d", "A1"])
def test_assert_safe_session_accepts(good):
    assert_safe_session(good)
    assert SAFE_SESSION_RE.match(good)


def test_local_disk_storage_rejects_traversal_session(tmp_path):
    config = connect_state(type="versioned", storage="disk", path=str(tmp_path))
    host = Local()
    with pytest.raises(ValueError, match="invalid session id"):
        host.resolve_state(config, "../escape")
    assert not (tmp_path.parent / "escape").exists()


def test_resolver_protocol_runtime_checkable():
    assert isinstance(BranchResolver(Memory()), StateResolver)
    assert not isinstance(object(), StateResolver)
