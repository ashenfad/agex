"""Tests for the silent-python nudge.

The loop emits a ``TASK_CONTROL_GUIDANCE`` ``OutputEvent`` on turns
where a ``python_action`` ran but produced no observable output — no
``print``, no ``view_image``, no raised error.  The nudge reminds the
model that returning normally is the implicit continue and that
finishing requires an explicit ``task_success`` / ``task_fail`` /
``task_clarify`` inside ``python_action``.

When the Python *did* produce output (stdout, an error, an image),
the agent can see that in its next tool_result and needs no reminder
— so the nudge should stay silent in those cases.
"""

import pytest

from agex import Agent, clear_agent_registry
from agex.agent.events import OutputEvent
from agex.eval.objects import PrintAction
from agex.llm import Dummy
from tests.agex._emissions import make_response

# Unique prefix from TASK_CONTROL_GUIDANCE — matching on it avoids
# newline-escaping issues that ``str(PrintAction)`` introduces.
_GUIDANCE_MARKER = "**Silent turn**"


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def _guidance_outputs(events):
    out = []
    for e in events:
        if not isinstance(e, OutputEvent):
            continue
        for part in e.parts:
            if isinstance(part, PrintAction) and any(
                _GUIDANCE_MARKER in str(arg) for arg in part.args
            ):
                out.append(e)
                break
    return out


def test_nudge_fires_on_silent_python():
    """Python ran but produced no stdout/image/error → one nudge."""
    llm = Dummy(
        responses=[
            make_response(thinking="silent", code="x = 1"),
            make_response(thinking="done", code="task_success('ok')"),
        ]
    )
    agent = Agent(name="quiet", llm=llm)

    @agent.task
    def run() -> str:
        """noop"""
        pass

    events: list = []
    assert run(on_event=events.append) == "ok"
    assert len(_guidance_outputs(events)) == 1


def test_nudge_silent_on_printed_python():
    """Python printed something → no nudge (output is observable)."""
    llm = Dummy(
        responses=[
            make_response(thinking="printing", code="print('hello')"),
            make_response(thinking="done", code="task_success('ok')"),
        ]
    )
    agent = Agent(name="loud", llm=llm)

    @agent.task
    def run() -> str:
        """noop"""
        pass

    events: list = []
    assert run(on_event=events.append) == "ok"
    assert _guidance_outputs(events) == []


def test_nudge_silent_on_raised_error():
    """Python raised → the error lands as an OutputEvent so the agent
    sees a concrete signal and the nudge stays silent."""
    llm = Dummy(
        responses=[
            make_response(thinking="oops", code="raise ValueError('bad')"),
            make_response(thinking="done", code="task_success('ok')"),
        ]
    )
    agent = Agent(name="oops", llm=llm)

    @agent.task
    def run() -> str:
        """noop"""
        pass

    events: list = []
    assert run(on_event=events.append) == "ok"
    assert _guidance_outputs(events) == []


def test_nudge_silent_on_terminator_call():
    """Explicit task_success — no silent-python condition, no nudge."""
    llm = Dummy(responses=[make_response(thinking="done", code="task_success(42)")])
    agent = Agent(name="term", llm=llm)

    @agent.task
    def run() -> int:
        """noop"""
        pass

    events: list = []
    assert run(on_event=events.append) == 42
    assert _guidance_outputs(events) == []
