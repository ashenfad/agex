"""Spike: scoped-but-ungranted stand-ins surface ``ScopeRequired`` through
the *real* sandtrap gates.

This validates the riskiest integration point of the scope-interrupt design
(IP-2): that we can register a raising stand-in which sandtrap permits, but
which raises an instructional error on use — for functions (call), classes
(construction), and modules (attribute access *and* ``from mod import x``).
"""

from sandtrap import Policy, Sandbox

from agex.eval.bridge.stubs import make_cls_stub, make_fn_stub, make_module_stub
from agex.eval.error import ScopeRequired


def _run(policy: Policy, code: str):
    return Sandbox(policy).exec(code, namespace={})


def test_fn_stub_raises_on_call():
    policy = Policy()
    policy.fn(make_fn_stub("send_mail", "email"), name="send_mail")

    result = _run(policy, "send_mail('hello', to='x@y.com')")

    assert isinstance(result.error, ScopeRequired)
    assert result.error.scope == "email"
    assert "task_request_permission(scope='email')" in str(result.error)


def test_fn_stub_referenceable_without_call():
    # The name resolves (agent can inspect it); only *calling* raises.
    policy = Policy()
    policy.fn(make_fn_stub("send_mail", "email"), name="send_mail")

    result = _run(policy, "f = send_mail")  # bind, don't call

    assert result.error is None


def test_cls_stub_raises_on_construction():
    policy = Policy()
    policy.cls(
        make_cls_stub("EmailClient", "email"),
        name="EmailClient",
        constructable=True,
    )

    result = _run(policy, "EmailClient()")

    assert isinstance(result.error, ScopeRequired)
    assert result.error.scope == "email"


def test_module_stub_raises_on_attr_access():
    policy = Policy()
    policy.module(make_module_stub("requests", "net"), name="requests", include="*")

    result = _run(policy, "requests.get('http://example.com')")

    assert isinstance(result.error, ScopeRequired)
    assert result.error.scope == "net"
    assert "requests.get" in str(result.error)


def test_module_stub_raises_on_from_import():
    policy = Policy()
    policy.module(make_module_stub("requests", "net"), name="requests", include="*")

    result = _run(policy, "from requests import get\nget('http://example.com')")

    assert isinstance(result.error, ScopeRequired)
    assert result.error.scope == "net"


def test_granted_real_member_unaffected():
    # Sanity: a real (non-stub) registration still works normally, proving the
    # stub behavior is specific to the stand-in, not the gate itself.
    policy = Policy()

    def real_send(msg: str) -> str:
        return f"sent: {msg}"

    policy.fn(real_send, name="send_mail")
    result = _run(policy, "out = send_mail('hi')")

    assert result.error is None
    assert result.namespace["out"] == "sent: hi"
