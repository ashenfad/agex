"""Milestone B (resume): Task.resume applies the host decision and re-enters
the suspended task — grant makes the scoped capability real, deny lets the
agent adapt.
"""

import pytest

from agex import Agent, clear_agent_registry, connect_state
from agex.agent.permission import PermissionPending
from agex.llm import Dummy
from tests.agex._emissions import make_response


def _agent(responses):
    clear_agent_registry()
    a = Agent(
        name="r",
        llm=Dummy(responses=responses),
        state=connect_state(type="versioned", storage="memory"),
    )

    @a.fn(scope="email")
    def send_mail(to):
        return f"sent to {to}"

    @a.task
    def chat(msg: str) -> str:
        """Chat."""

    return a, chat


def _suspend(chat, session):
    try:
        chat("hi", session=session)
        raise AssertionError("expected PermissionPending")
    except PermissionPending as p:
        return p


def test_resume_grant_round_trip():
    a, chat = _agent(
        [
            make_response(
                thinking="need email",
                code="task_request_permission('email', reason='to send')",
            ),
            make_response(thinking="now send", code="task_success(send_mail('boss'))"),
        ]
    )
    p = _suspend(chat, "s")
    assert p.scope == "email"

    # Grant and resume: the scoped fn is now real, so the task completes.
    result = chat.resume(session="s", response=p.respond(granted=True))
    assert result == "sent to boss"


def test_resume_deny_round_trip():
    a, chat = _agent(
        [
            make_response(code="task_request_permission('email')"),
            make_response(code="task_success('gave up: no email access')"),
        ]
    )
    p = _suspend(chat, "d")

    # Deny: the agent sees the denial and adapts.
    result = chat.resume(
        session="d", response=p.respond(granted=False, note="not allowed")
    )
    assert result == "gave up: no email access"


def test_full_hitl_loop_via_stub():
    # The natural flow: the agent *tries* the scoped capability, hits the
    # ScopeRequired stub (an observation), then asks; after a grant + resume
    # the real capability works.
    a, chat = _agent(
        [
            make_response(code="x = send_mail('boss')"),  # ScopeRequired observation
            make_response(
                code="task_request_permission('email', reason='to send')"
            ),  # asks
            make_response(code="task_success(send_mail('boss'))"),  # after resume
        ]
    )
    p = _suspend(chat, "n")
    assert p.scope == "email"

    result = chat.resume(session="n", response=p.respond(granted=True))
    assert result == "sent to boss"


def test_resume_with_no_open_request_errors():
    a, chat = _agent([make_response(code="task_success('done')")])
    # No suspension happened; resume should refuse.
    chat("hi", session="x")  # completes normally
    try:
        from agex.agent.permission import PermissionResponse

        chat.resume(session="x", response=PermissionResponse(granted=True))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "no open permission request" in str(e).lower()


def test_resume_re_suspends_on_second_scope():
    # A resumed turn that asks for *another* scope raises PermissionPending
    # again (the loop).
    clear_agent_registry()
    a = Agent(
        name="r2",
        llm=Dummy(
            responses=[
                make_response(code="task_request_permission('email')"),
                make_response(code="task_request_permission('net')"),
            ]
        ),
        state=connect_state(type="versioned", storage="memory"),
    )

    @a.task
    def chat(msg: str) -> str:
        """Chat."""

    p = _suspend(chat, "s2")
    assert p.scope == "email"
    try:
        chat.resume(session="s2", response=p.respond(granted=True))
        raise AssertionError("expected second PermissionPending")
    except PermissionPending as p2:
        assert p2.scope == "net"


@pytest.mark.asyncio
async def test_async_resume_round_trip():
    clear_agent_registry()
    a = Agent(
        name="ar",
        llm=Dummy(
            responses=[
                make_response(
                    code="task_request_permission('email', reason='to send')"
                ),
                make_response(code="task_success(send_mail('boss'))"),
            ]
        ),
        state=connect_state(type="versioned", storage="memory"),
    )

    @a.fn(scope="email")
    def send_mail(to):
        return f"sent to {to}"

    @a.task
    async def chat(msg: str) -> str:
        """Chat."""

    with pytest.raises(PermissionPending) as ei:
        await chat("hi", session="as")
    assert ei.value.scope == "email"

    result = await chat.aresume(session="as", response=ei.value.respond(granted=True))
    assert result == "sent to boss"
