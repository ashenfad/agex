"""The tools a session offers: the workspace's own by default, shaped by
an embedder's ``toolset`` and ``tools`` (MCP's shape: name,
description, a JSON Schema, a call returning a ``ToolOutput``)."""

import pytest
from nontainer import Store
from nontainer.adapters.tools import Tool, ToolOutput, Toolset
from nontainer.conformance.corpus import calls, says

from agex import Agent
from agex.providers.scripted import ScriptedProvider


@pytest.fixture
def ws():
    store = Store(memory=True)
    ws = store.open("s")
    yield ws
    ws.close()
    store.close()


def agent(*steps):
    return Agent(ScriptedProvider(list(steps)))


def offered(session) -> dict[str, str]:
    """What the model is offered: each tool's name and description."""
    return {spec.name: spec.description for spec in session._specs}


LOOKUP = Tool(
    name="lookup",
    description="A student's name as the directory files it.",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    call=lambda key: ToolOutput(text=key.upper()),
)


def result_of(session) -> str:
    (result,) = session.runs[-1].messages[2].parts
    return result.content


def test_by_default_a_session_offers_the_workspaces_own_tools(ws):
    session = agent(says("ok")).session(ws)
    assert set(offered(session)) == {
        "terminal",
        "file_write",
        "file_edit",
        "run_python",
    }


def test_a_tool_of_the_embedders_is_offered_and_called(ws):
    session = agent(calls("lookup", key="ada"), says("found")).session(
        ws, tools=[LOOKUP]
    )
    assert offered(session)["lookup"] == LOOKUP.description
    outcome = session.say("look ada up")
    assert outcome.status == "completed"
    assert result_of(session) == "ADA"


def test_a_tool_named_like_a_built_in_replaces_it(ws):
    terminal = Tool(
        name="terminal",
        description="The embedder's terminal.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        call=lambda command: ToolOutput(text=f"ran {command} elsewhere"),
    )
    session = agent(calls("terminal", command="ls"), says("done")).session(
        ws, tools=[terminal]
    )
    assert offered(session)["terminal"] == "The embedder's terminal."
    session.say("list")
    assert result_of(session) == "ran ls elsewhere"


def test_an_embedders_toolset_shapes_the_built_ins(ws):
    toolset = Toolset(ws, python_primer="Prefer polars.", vision=True)
    session = agent(says("ok")).session(ws, toolset=toolset)
    tools = offered(session)
    assert "view_image" in tools  # the embedder's model takes images
    assert "Prefer polars." in tools["run_python"]
    assert session.toolset is toolset


def test_an_embedders_toolset_takes_a_built_sessions_not_true(ws):
    with pytest.raises(ValueError, match="sessions=True"):
        agent().session(ws, toolset=Toolset(ws), sessions=True)


def test_an_embedders_toolset_delivers_through_the_sessions_it_is_given(ws):
    """The studio's shape: its own toolset (no sessions tool of
    nontainer's), its own sessions tool among ``tools``, and the helper
    it delegates through, whose answers the session delivers."""
    import asyncio

    from nontainer.sessions import Sessions

    from agex.delegation import Runner

    a = agent(says("ok"))
    loop = asyncio.new_event_loop()
    helper = Sessions(ws, Runner(a, ws), loop=loop)
    try:
        session = a.session(ws, toolset=Toolset(ws), sessions=helper)
        assert session.sessions is helper
        assert "sessions" not in offered(session)
    finally:
        helper.close()
        loop.close()
