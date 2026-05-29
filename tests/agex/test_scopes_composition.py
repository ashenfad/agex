"""Milestone C — composition guard (sibling constraint) and the conditional,
cache-safe permission primer section.
"""

from agex import Agent, clear_agent_registry


def test_scoped_agent_cannot_be_registered_as_sub_agent():
    clear_agent_registry()
    specialist = Agent(name="spec")

    @specialist.fn(scope="email")
    def send_mail(to): ...

    @specialist.task
    def do_email(x: str) -> str:
        """Email task."""

    orchestrator = Agent(name="orch")
    try:
        orchestrator.fn(do_email)
        raise AssertionError("expected ValueError for scoped sub-agent")
    except ValueError as e:
        msg = str(e).lower()
        assert "scoped" in msg and "sub-agent" in msg


def test_unscoped_agent_can_be_registered_as_sub_agent():
    clear_agent_registry()
    specialist = Agent(name="spec2")

    @specialist.task
    def helper(x: str) -> str:
        """Helper."""

    orchestrator = Agent(name="orch2")
    orchestrator.fn(helper)  # must not raise
    assert "helper" in orchestrator._policy.namespaces["__main__"].fn_objects


def test_primer_includes_permission_section_when_scoped():
    clear_agent_registry()
    a = Agent(name="p1")

    @a.fn(scope="email")
    def send_mail(to): ...

    a.module(__import__("math"), scope="net")

    msg = a._build_system_message()
    assert "# Permission Scopes" in msg
    assert "task_request_permission" in msg
    # The scope vocabulary is listed (static, cache-safe).
    assert "email" in msg and "net" in msg


def test_primer_omits_permission_section_without_scopes():
    clear_agent_registry()
    a = Agent(name="p2")

    @a.fn
    def free(): ...

    msg = a._build_system_message()
    assert "Permission Scopes" not in msg
    assert "task_request_permission" not in msg
