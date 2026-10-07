"""A session: an agent driving a workspace turn by turn, its
conversation stored in the workspace's branch."""

import asyncio

import pytest
from nontainer import NotSupportedError, Store, TurnInProgress, conversation
from nontainer.conformance.corpus import calls, fails, says, writes
from nontainer.conversation import Index
from nontainer.turns import (
    Delivered,
    RunEnded,
    RunStarted,
    TextDelta,
    ToolEnded,
    ToolStarted,
    Usage,
)
from pydantic_ai import messages as pai

from agex import Agent, Outcome
from agex.agent import CLOSING_NOTE, HARNESS
from agex.providers.scripted import ScriptedProvider
from agex.record import Text, ToolCall, ToolResult


@pytest.fixture
def ws():
    store = Store(memory=True)
    ws = store.open("s")
    yield ws
    ws.close()
    store.close()


def agent(*steps, **kw):
    return Agent(ScriptedProvider(list(steps)), **kw)


def kinds(events):
    return [type(e).__name__ for e in events if not isinstance(e, Usage)]


def test_a_turn_replies_and_lands_one_commit(ws):
    head = ws.head
    outcome = agent(says("hello")).session(ws).say("say hello")
    assert isinstance(outcome, Outcome)
    assert (outcome.status, outcome.text) == ("completed", "hello")
    assert outcome.head == ws.head != head
    assert kinds(outcome.events) == ["RunStarted", "TextDelta", "RunEnded"]
    assert ws.log(limit=1)[0].info == {
        "tool": "turn",
        "runs": {outcome.run_id: "completed"},
    }
    index = conversation.index_of(ws)
    assert (index.harness, index.runs) == (HARNESS, (outcome.run_id,))


def test_tools_run_against_the_workspace_and_the_run_keeps_them(ws):
    session = agent(writes("/workspace/a.txt", "A"), says("wrote it")).session(ws)
    outcome = session.say("write a")
    assert ws.files.read("/workspace/a.txt") == b"A"
    assert kinds(outcome.events) == [
        "RunStarted",
        "ToolStarted",
        "ToolEnded",
        "TextDelta",
        "RunEnded",
    ]
    started = next(e for e in outcome.events if isinstance(e, ToolStarted))
    ended = next(e for e in outcome.events if isinstance(e, ToolEnded))
    assert started.call_id == ended.call_id == "call_1"
    assert started.args == {"path": "/workspace/a.txt", "content": "A"}
    assert not ended.is_error
    assert [e.info.get("tool") for e in reversed(ws.log(limit=2))] == [
        "file_write",
        "turn",
    ]
    (run,) = session.runs
    assert [m.role for m in run.messages] == ["user", "assistant", "tool", "assistant"]
    assert run.status == "completed" and run.started_at and run.ended_at
    (call,) = run.messages[1].parts
    (result,) = run.messages[2].parts
    assert isinstance(call, ToolCall) and isinstance(result, ToolResult)
    assert result.call_id == call.call_id and not result.is_error


def test_the_conversation_carries_from_turn_to_turn(ws):
    provider = ScriptedProvider([says("one"), says("two")])
    session = Agent(provider, primer="You are terse.").session(ws)
    session.say("first")
    session.say("second")
    sent = provider.seen[1]
    (request,) = [m for m in sent[:1]]
    system, user = request.parts
    assert isinstance(system, pai.SystemPromptPart)
    assert system.content.startswith("You are terse.")
    assert "workspace" in system.content  # the tool instructions follow
    assert user.content == "first"
    assert [type(m).__name__ for m in sent] == [
        "ModelRequest",
        "ModelResponse",
        "ModelRequest",
    ]
    assert sent[2].parts[0].content == "second"
    assert [r.messages[0].text for r in session.runs] == ["first", "second"]


def test_a_new_session_continues_the_stored_conversation(ws):
    agent(says("noted")).session(ws).say("remember 42")
    provider = ScriptedProvider([says("42")])
    Agent(provider).session(ws).say("what was it?")
    texts = [
        part.content
        for message in provider.seen[0]
        for part in message.parts
        if isinstance(part, (pai.UserPromptPart, pai.TextPart))
    ]
    assert texts == ["remember 42", "noted", "what was it?"]


def test_stream_yields_events_as_they_happen_and_leaves_the_outcome(ws):
    session = agent(says("streamed")).session(ws)

    async def go():
        return [e async for e in session.stream("go")]

    events = asyncio.run(go())
    assert isinstance(events[0], RunStarted) and isinstance(events[-1], RunEnded)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "streamed"
    assert session.last is not None and session.last.text == "streamed"
    assert any(isinstance(e, Usage) and e.input_tokens > 0 for e in events)


def test_a_note_queued_mid_turn_rides_the_next_tool_result(ws):
    session = agent(writes("/workspace/a.txt", "A"), says("ok")).session(ws)
    note = session.inbox.put("also mention the date")
    outcome = session.say("write a")
    assert kinds(outcome.events) == [
        "RunStarted",
        "ToolStarted",
        "Delivered",
        "ToolEnded",
        "TextDelta",
        "RunEnded",
    ]
    (delivered,) = [e for e in outcome.events if isinstance(e, Delivered)]
    assert [n.id for n in delivered.notes] == [note.id]
    ended = next(e for e in outcome.events if isinstance(e, ToolEnded))
    assert "also mention the date" not in ended.result
    (result,) = session.runs[0].messages[2].parts
    assert "also mention the date" in result.content  # what the model read
    assert session.inbox.pending() == [] and session.inbox.delivered() == []


def test_a_failed_tool_call_is_a_result_the_model_reads(ws):
    session = agent(
        calls("no_such_tool"),
        calls("file_write", path="/workspace/a.txt"),  # content missing
        says("I see"),
    ).session(ws)
    outcome = session.say("go")
    assert outcome.status == "completed"
    ended = [e for e in outcome.events if isinstance(e, ToolEnded)]
    assert [e.is_error for e in ended] == [True, True]
    assert "no tool named 'no_such_tool'" in ended[0].result
    assert "TypeError" in ended[1].result


def test_a_run_that_never_stops_calling_tools_is_stopped(ws):
    session = agent(*[calls("terminal", command="true")] * 5, max_steps=3).session(ws)
    outcome = session.say("go")
    assert outcome.status == "failed" and "3 calls" in outcome.message
    assert ws.log(limit=1)[0].info["runs"] == {outcome.run_id: "failed"}


def test_leaving_the_stream_early_stops_watching_and_the_turn_goes_on(ws):
    """A plain ``break`` closes nothing: the turn is the session's, so it
    runs to its end, lands its commit, and leaves the session free."""
    session = agent(writes("/workspace/a.txt", "A"), says("done")).session(ws)

    async def go():
        stream = session.stream("write a")
        async for event in stream:
            assert isinstance(event, RunStarted)
            break
        return await stream.wait()

    outcome = asyncio.run(go())
    assert (outcome.status, outcome.text) == ("completed", "done")
    assert ws.files.read("/workspace/a.txt") == b"A"
    assert ws.turns.current is None and not session.running
    assert session.runs[0].status == "completed"


def test_an_abandoned_stream_does_not_hold_the_session(ws):
    """A stream left mid-turn and never closed: its turn still ends on
    its own, and the next turn is free to start."""
    session = agent(says("first"), says("second")).session(ws)

    async def go():
        held = session.stream("one")
        await held.__anext__()  # watch the start, then stop watching
        for _ in range(1000):
            if not session.running:
                break
            await asyncio.sleep(0)
        assert not session.running
        return held, await session.asay("two")

    held, second = asyncio.run(go())
    assert held.outcome is not None and held.outcome.text == "first"
    assert second.text == "second"
    assert [run.status for run in session.runs] == ["completed", "completed"]


def test_one_turn_at_a_time(ws):
    session = agent(says("a"), says("b")).session(ws)

    async def go():
        first = session.stream("one")
        await first.__anext__()
        with pytest.raises(TurnInProgress):
            await session.stream("two").__anext__()
        return await first.wait()

    assert asyncio.run(go()).text == "a"


def test_an_error_ends_the_turn_failed_and_keeps_its_work(ws):
    """No ending raises: the stream closes with RunEnded, the outcome
    says why, and the run keeps what it did with a closing note."""
    session = agent(writes("/workspace/a.txt", "A"), fails("error")).session(ws)

    async def go():
        stream = session.stream("go")
        events = [e async for e in stream]
        return events, await stream.wait()

    events, outcome = asyncio.run(go())
    assert isinstance(events[-1], RunEnded) and events[-1].status == "failed"
    assert outcome.status == "failed" and "scripted model failed" in outcome.message
    (run,) = session.runs
    assert run.status == "failed"
    assert run.messages[-1].text.startswith(CLOSING_NOTE)
    assert "scripted model failed" in run.messages[-1].text
    assert ws.files.read("/workspace/a.txt") == b"A"
    assert ws.log(limit=1)[0].info["runs"] == {outcome.run_id: "failed"}


def test_say_refuses_to_block_a_running_loop(ws):
    session = agent(says("x")).session(ws)

    async def go():
        with pytest.raises(RuntimeError, match="asay"):
            session.say("go")
        return await session.asay("go")

    assert asyncio.run(go()).text == "x"


def test_another_harnesss_conversation_is_refused(ws):
    conversation.write(ws.provider.kv, Index(harness="agno", session="s"))
    ws.commit(info={"tool": "test"})
    with pytest.raises(NotSupportedError, match="agno"):
        agent().session(ws)


def test_a_model_name_becomes_a_pydantic_ai_provider():
    assert Agent("test").provider.name == "test:test"


def test_the_run_record_holds_text_parts(ws):
    outcome = agent(says("hi")).session(ws).say("hello")
    body = conversation.read_runs(ws.provider.kv, [outcome.run_id])[outcome.run_id]
    assert body["run_id"] == outcome.run_id and body["status"] == "completed"
    assert body["messages"][0]["parts"] == [{"kind": "text", "text": "hello"}]
    assert Text(text="hello") in agent().session(ws).runs[0].messages[0].parts
