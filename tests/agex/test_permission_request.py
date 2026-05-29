"""Milestone B (suspend): ``task_request_permission`` suspends the task,
committing a ``PermissionRequestEvent`` and raising ``PermissionPending`` to
the host.
"""

import pytest

from agex import Agent, clear_agent_registry, connect_state, events
from agex.agent.events import PermissionRequestEvent
from agex.agent.permission import PermissionPending, PermissionResponse
from agex.llm import Dummy
from tests.agex._emissions import make_response


def _agent():
    clear_agent_registry()
    a = Agent(
        name="b", llm=Dummy(), state=connect_state(type="versioned", storage="memory")
    )

    @a.task
    def chat(msg: str) -> str:
        """Chat."""

    return a, chat


def test_request_permission_suspends_with_pending_and_event():
    a, chat = _agent()
    a.llm.responses = [
        make_response(
            thinking="need to send mail",
            code="task_request_permission('email', reason='to send the summary')",
        )
    ]

    try:
        chat("hi", session="s")
        raise AssertionError("expected PermissionPending")
    except PermissionPending as p:
        assert p.scopes == {"email"}
        assert p.task_name == "chat"
        assert p.reason == "to send the summary"
        resp = p.respond(granted=True, note="ok")
        assert isinstance(resp, PermissionResponse)
        assert resp.granted is True and resp.note == "ok"

    # The request is durably recorded in the session log.
    evs = events(a.state("s"))
    reqs = [e for e in evs if isinstance(e, PermissionRequestEvent)]
    assert len(reqs) == 1
    assert reqs[0].scopes == {"email"}
    assert reqs[0].task_name == "chat"
    assert reqs[0].reason == "to send the summary"


def test_request_permission_without_reason():
    a, chat = _agent()
    a.llm.responses = [
        make_response(thinking="ask", code="task_request_permission('net')")
    ]
    try:
        chat("hi", session="s")
        raise AssertionError("expected PermissionPending")
    except PermissionPending as p:
        assert p.scopes == {"net"}
        assert p.reason is None


@pytest.mark.asyncio
async def test_async_request_permission_suspends():
    clear_agent_registry()
    a = Agent(
        name="ab",
        llm=Dummy(
            responses=[
                make_response(
                    thinking="ask",
                    code="task_request_permission('email', reason='r')",
                )
            ]
        ),
        state=connect_state(type="versioned", storage="memory"),
    )

    @a.task
    async def achat(msg: str) -> str:
        """Chat."""

    with pytest.raises(PermissionPending) as ei:
        await achat("hi", session="s")
    assert ei.value.scopes == {"email"}
    assert ei.value.reason == "r"

    reqs = [e for e in events(a.state("s")) if isinstance(e, PermissionRequestEvent)]
    assert len(reqs) == 1
    assert reqs[0].scopes == {"email"}
