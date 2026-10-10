"""Agent-defined tasks against a real model: one round trip with agex
driving the lead, and one with agno driving it through nontainer's
toolkit, each world granted ``agex`` the same way.

Opt-in, as the other live tests (``-m live``), on Anthropic's small
model (``AGEX_LIVE_ANTHROPIC`` overrides it). The agno one needs agno,
which agex doesn't depend on:

    uv run --with agno --all-extras python -m pytest -m live tests/test_live_agent_tasks.py
"""

import os

import pytest
from nontainer import Profile, PythonConfig, Store

from agex import Agent
from agex.agent_tasks import agent_tasks

pytestmark = pytest.mark.live

MODEL = os.environ.get("AGEX_LIVE_ANTHROPIC", "anthropic:claude-haiku-4-5-20251001")

PROMPT = """\
In run_python, do this in one script:
1. Define a dataclass Point2D with fields x: float and y: float.
2. Define a task with the `agex` object you hold:

    @agex.task
    def corners(shape: str) -> list[Point2D]:
        \"""The corner points of the named shape, counterclockwise from the origin.\"""

3. Call corners("unit square") and print len(result), and whether each item
   is a Point2D (isinstance).
Then reply with the number of points it printed, and nothing else."""


@pytest.fixture
def key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("needs ANTHROPIC_API_KEY")


def world(store, helpers):
    return store.open(
        "lead",
        profile=Profile(
            python=PythonConfig(host_objects={"agex": agent_tasks(helpers)})
        ),
    )


def test_an_agex_lead_defines_and_calls_a_task(key):
    agent = Agent(MODEL, primer="You are terse.")
    store = Store(memory=True)
    ws = world(store, agent)
    chat = agent.session(ws, sessions=True)
    try:
        outcome = chat.say(PROMPT)
        if outcome.status == "interrupted":
            outcome = chat.resume()
        assert outcome.status == "completed", outcome.message
        assert "4" in outcome.text, outcome.text
        jobs = chat.sessions.list()
        assert jobs and all(j.name.startswith("lead.corners-") for j in jobs), jobs
        assert any(j.status == "answered" for j in jobs), jobs
    finally:
        chat.close()
        ws.close()
        store.close()


def test_an_agno_lead_defines_and_calls_a_task(key):
    agno_agent = pytest.importorskip("agno.agent")
    claude = pytest.importorskip("agno.models.anthropic")
    from nontainer.adapters.agno import WorkspaceTools

    helpers = Agent(MODEL, primer="You are terse.")
    store = Store(memory=True)
    ws = world(store, helpers)
    try:
        lead = agno_agent.Agent(
            model=claude.Claude(id=MODEL.split(":", 1)[1]),
            tools=[WorkspaceTools(ws)],
            instructions="You are terse.",
        )
        run = lead.run(PROMPT)
        assert "4" in str(run.content), run.content
        printed = [
            str(getattr(m, "content", ""))
            for m in run.messages or []
            if getattr(m, "role", None) == "tool"
        ]
        assert any("True" in p for p in printed), printed
    finally:
        ws.close()
        store.close()
