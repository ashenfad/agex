"""Milestone A — capability gating.

Covers ``scope=`` on registrations, ``agent.scope_names``, the
``scopes(state)`` grant accessor, and the per-execution effective-policy
overlay that substitutes a ``ScopeRequired`` stand-in for a
scoped-but-ungranted registration.
"""

import math

from sandtrap import Sandbox

from agex import Agent, clear_agent_registry, connect_state, events, scopes
from agex.agent.events import PermissionEvent
from agex.eval.bridge.policy import translate_policy
from agex.eval.error import ScopeRequired
from agex.llm import Dummy
from agex.state.live import Live
from agex.state.scopes import read_grants
from tests.agex._emissions import make_response

# --- scope_names -----------------------------------------------------------


def test_scope_names_collects_fn_cls_module():
    a = Agent()

    @a.fn(scope="email")
    def send_mail(to): ...

    @a.cls(scope="email")
    class EmailClient: ...

    a.module(math, scope="net")

    @a.fn
    def free(): ...

    assert a.scope_names == {"email", "net"}


# --- scopes(state) accessor ------------------------------------------------


def test_scopes_accessor_on_live():
    state = Live()
    s = scopes(state)
    assert s.list() == set()
    assert not s.has("email")
    s.grant("email")
    s.grant("email")  # idempotent
    assert s.has("email")
    assert s.list() == {"email"}
    s.revoke("email")
    s.revoke("email")  # idempotent
    assert not s.has("email")
    assert s.list() == set()


def test_scopes_accessor_versioned_persists():
    clear_agent_registry()
    agent = Agent(
        name="g", llm=Dummy(), state=connect_state(type="versioned", storage="memory")
    )

    @agent.task
    def noop() -> None:
        """Do nothing."""

    agent.llm.responses = [make_response(thinking="done", code="task_success(None)")]
    noop(session="s")

    scopes(agent.state("s")).grant("email")

    # A fresh handle reflects the committed grant.
    assert read_grants(agent.state("s")) == {"email"}
    assert scopes(agent.state("s")).has("email")

    scopes(agent.state("s")).revoke("email")
    assert read_grants(agent.state("s")) == set()


def test_scopes_accessor_emits_permission_event():
    clear_agent_registry()
    agent = Agent(
        name="ge", llm=Dummy(), state=connect_state(type="versioned", storage="memory")
    )

    @agent.task
    def noop() -> None:
        """Do nothing."""

    agent.llm.responses = [make_response(thinking="done", code="task_success(None)")]
    noop(session="s")

    # grant → a PermissionEvent(granted=...) by the host ("System")
    scopes(agent.state("s")).grant("email")
    grants = [
        e
        for e in events(agent.state("s"))
        if isinstance(e, PermissionEvent) and e.granted
    ]
    assert len(grants) == 1
    assert grants[0].granted == ["email"]
    assert grants[0].agent_name == "System"

    # revoke → a PermissionEvent(revoked=...)
    scopes(agent.state("s")).revoke("email")
    revokes = [
        e
        for e in events(agent.state("s"))
        if isinstance(e, PermissionEvent) and e.revoked
    ]
    assert len(revokes) == 1
    assert revokes[0].revoked == ["email"]

    # idempotent: granting an already-granted scope emits nothing new
    scopes(agent.state("s")).grant("email")
    scopes(agent.state("s")).grant("email")  # no-op
    grant_events = [
        e
        for e in events(agent.state("s"))
        if isinstance(e, PermissionEvent) and e.granted == ["email"]
    ]
    assert len(grant_events) == 2  # first grant + the re-grant after revoke


# --- effective-policy overlay ----------------------------------------------


def _agent_with_scoped_fn() -> Agent:
    a = Agent()

    @a.fn(scope="email")
    def send_mail(to):
        return f"sent to {to}"

    return a


def test_overlay_ungranted_fn_is_stub():
    a = _agent_with_scoped_fn()
    policy = translate_policy(a, grants=set())
    result = Sandbox(policy).exec("out = send_mail('x@y.com')", namespace={})
    assert isinstance(result.error, ScopeRequired)
    assert result.error.scope == "email"


def test_overlay_granted_fn_is_real():
    a = _agent_with_scoped_fn()
    policy = translate_policy(a, grants={"email"})
    result = Sandbox(policy).exec("out = send_mail('x@y.com')", namespace={})
    assert result.error is None
    assert result.namespace["out"] == "sent to x@y.com"


def test_overlay_none_grants_locks_scoped():
    # grants=None must be treated as "nothing granted".
    a = _agent_with_scoped_fn()
    policy = translate_policy(a, grants=None)
    result = Sandbox(policy).exec("out = send_mail('x')", namespace={})
    assert isinstance(result.error, ScopeRequired)


def test_overlay_unscoped_always_available():
    a = Agent()

    @a.fn
    def ping():
        return "pong"

    policy = translate_policy(a, grants=set())
    result = Sandbox(policy).exec("out = ping()", namespace={})
    assert result.error is None
    assert result.namespace["out"] == "pong"


def test_overlay_scoped_module_gated_then_granted():
    a = Agent()
    a.module(math, scope="mathscope")

    gated = Sandbox(translate_policy(a, grants=set())).exec(
        "out = math.sqrt(4)", namespace={}
    )
    assert isinstance(gated.error, ScopeRequired)

    granted = Sandbox(translate_policy(a, grants={"mathscope"})).exec(
        "out = math.sqrt(4)", namespace={}
    )
    assert granted.error is None
    assert granted.namespace["out"] == 2.0
