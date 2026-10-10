"""The tools a session offers: the workspace's own by default, shaped by
an embedder's ``toolset`` and ``tools`` (MCP's shape: name,
description, a JSON Schema, a call returning a ``ToolOutput``)."""

import pytest
from nontainer import Store
from nontainer.adapters.tools import Tool, ToolImage, ToolOutput, Toolset
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


def png() -> bytes:
    """A 1x1 PNG."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00\xff\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )


def sent_images(provider) -> list:
    """The images the last request carried in tool results."""
    from pydantic_ai import messages as pai

    return [
        c
        for m in provider.seen[-1]
        for p in getattr(m, "parts", ())
        if isinstance(p, pai.ToolReturnPart) and isinstance(p.content, list)
        for c in p.content
        if isinstance(c, pai.BinaryContent)
    ]


def test_an_image_a_tool_returns_reaches_a_model_that_takes_images(ws):
    """view_image's image goes in its tool result, in the next request,
    and the run keeps it: a resumed or later turn sends it again."""
    from agex.record import dump_run, load_run

    ws.files.write("/workspace/dot.png", png())
    provider = ScriptedProvider(
        [calls("view_image", path="/workspace/dot.png"), says("a green dot")]
    )
    session = Agent(provider).session(ws, toolset=Toolset(ws, vision=True))
    assert session.say("look at it").status == "completed"
    (image,) = sent_images(provider)
    assert (image.data, image.media_type) == (png(), "image/png")
    (result,) = session.runs[-1].messages[2].parts
    assert result.images and result.images[0].media_type == "image/png"
    run = session.runs[-1]
    assert load_run(dump_run(run)) == run


SNAP = Tool(
    name="snap",
    description="A screenshot.",
    parameters={"type": "object", "properties": {}},
    call=lambda: ToolOutput(
        text="saved /workspace/shot.png",
        images=(ToolImage(data=png(), format="png"),),
    ),
)


def test_a_model_that_takes_no_images_is_never_sent_one(ws):
    provider = ScriptedProvider([calls("snap"), says("ok")])
    session = Agent(provider).session(ws, tools=[SNAP])  # vision off
    session.say("take one")
    assert sent_images(provider) == []
    (result,) = session.runs[-1].messages[2].parts
    assert result.images == () and result.content == "saved /workspace/shot.png"
