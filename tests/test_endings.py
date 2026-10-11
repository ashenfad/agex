"""How a turn ends: cancelled, failed, interrupted then resumed. None of
the endings raise, every ending lands the run, and a cancelled or failed
run keeps its work with a closing note."""

import asyncio
import threading
import time

import pytest
from nontainer import Store
from nontainer.conformance.corpus import fails, says, writes
from nontainer.turns import RunEnded, RunStarted, ToolEnded, ToolStarted
from pydantic_ai.models.function import FunctionModel

from agex import Agent
from agex.agent import CLOSING_NOTE, CUT_OFF, STOPPED, UNANSWERED
from agex.providers.pydanticai import PydanticAIProvider
from agex.providers.scripted import ScriptedProvider
from agex.record import ToolCall, ToolResult


@pytest.fixture
def ws():
    store = Store(memory=True)
    ws = store.open("s")
    yield ws
    ws.close()
    store.close()


def agent(*steps, **kw):
    return Agent(ScriptedProvider(list(steps)), **kw)


class Slow:
    """A model that takes its time: each call waits until released (or
    cancelled), then says what it was given."""

    def __init__(self, text="slowly"):
        self.text = text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def provider(self):
        async def reply(messages, info):
            self.started.set()
            await self.release.wait()
            yield self.text

        return PydanticAIProvider(
            FunctionModel(stream_function=reply, model_name="slow")
        )


# -- interrupted, then resumed ---------------------------------------------------


def test_a_provider_error_interrupts_and_resume_continues_in_place(ws):
    session = agent(
        writes("/workspace/a.txt", "A"),
        fails("provider"),
        writes("/workspace/b.txt", "B"),
        says("both written"),
    ).session(ws)
    first = session.say("write a and b")
    assert first.status == "interrupted" and "529" in first.message
    (run,) = session.runs
    assert run.status == "interrupted"
    assert not run.messages[-1].text.startswith(CLOSING_NOTE)

    second = session.resume()
    assert (second.status, second.text, second.run_id) == (
        "completed",
        "both written",
        first.run_id,
    )
    (run,) = session.runs  # still one run, continued
    results = [p for m in run.messages for p in m.parts if isinstance(p, ToolResult)]
    assert len(results) == 2
    assert [m.role for m in run.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [e.info["runs"] for e in ws.log() if e.info.get("tool") == "turn"] == [
        {first.run_id: "completed"},
        {first.run_id: "interrupted"},
    ]


def test_only_an_interrupted_last_turn_resumes(ws):
    session = agent(says("done"), fails("error")).session(ws)
    with pytest.raises(ValueError, match="never ran"):
        session.resume()
    session.say("go")
    with pytest.raises(ValueError, match="ended completed"):
        session.resume()
    session.say("again")  # fails, which is not resumable
    with pytest.raises(ValueError, match="ended failed"):
        session.resume()
    with pytest.raises(
        ValueError, match="one or the other|prompt, resumes the last one, or is woken"
    ):
        session.stream("both", resume=True)


def test_an_abandoned_interruption_ends_with_its_note(ws):
    provider = ScriptedProvider(
        [writes("/workspace/a.txt", "A"), fails("provider"), says("ok")]
    )
    session = Agent(provider).session(ws)
    first = session.say("write a")
    run = session.abandon("the provider failed 4 times in a row")
    assert run.status == "failed"
    assert run.messages[-1].role == "assistant"
    assert run.messages[-1].text.startswith(CLOSING_NOTE)
    assert "failed 4 times in a row" in run.messages[-1].text
    # stored: one run, no longer resumable, the commit saying how it ended
    assert session.runs == [run]
    with pytest.raises(ValueError, match="ended failed"):
        session.resume()
    assert [e.info for e in ws.log() if e.info.get("tool") == "turn"][0] == {
        "tool": "turn",
        "runs": {first.run_id: "failed"},
        "message": "the provider failed 4 times in a row",
    }
    # the next turn's model reads the work and the note
    assert session.say("go on").status == "completed"
    sent = [
        part.content
        for m in provider.seen[-1]
        for part in m.parts
        if isinstance(getattr(part, "content", None), str)
    ]
    assert any("failed 4 times in a row" in text for text in sent)


def test_an_abandoned_interruption_can_end_cancelled(ws):
    session = agent(fails("provider")).session(ws)
    session.say("go")
    run = session.abandon("stopped by the user", status="cancelled")
    assert session.runs[-1].status == run.status == "cancelled"
    assert "stopped by the user" in run.messages[-1].text


def test_only_an_interrupted_last_turn_is_abandoned(ws):
    session = agent(says("done"), fails("provider")).session(ws)
    with pytest.raises(ValueError, match="never ran"):
        session.abandon("why")
    session.say("go")
    with pytest.raises(ValueError, match="ended completed"):
        session.abandon("why")
    session.say("again")
    with pytest.raises(ValueError, match="not 'interrupted'"):
        session.abandon("why", status="interrupted")  # type: ignore[arg-type]
    assert session.runs[-1].status == "interrupted"


def test_a_new_turn_after_an_interruption_carries_its_work(ws):
    provider = ScriptedProvider(
        [writes("/workspace/a.txt", "A"), fails("provider"), says("ok")]
    )
    session = Agent(provider).session(ws)
    session.say("write a")
    assert session.say("never mind").status == "completed"
    sent = provider.seen[-1]
    assert sum(len(m.parts) for m in sent) >= 4  # the interrupted run is history


# -- cancelled -------------------------------------------------------------------


def test_cancel_stops_a_model_call_in_flight_and_returns_the_outcome(ws):
    slow = Slow()
    session = Agent(slow.provider()).session(ws)

    async def go():
        stream = session.stream("take your time")
        waiting = asyncio.ensure_future(stream.wait())
        await slow.started.wait()
        assert session.cancel() is True
        return [e async for e in stream], await waiting

    events, outcome = asyncio.run(go())
    assert (outcome.status, outcome.message) == ("cancelled", STOPPED)
    assert isinstance(events[0], RunStarted) and events[-1].status == "cancelled"
    (run,) = session.runs
    assert run.status == "cancelled" and run.messages[-1].text.startswith(CLOSING_NOTE)
    assert not session.running and session.cancel() is False


def test_a_cancel_says_why_in_the_closing_note(ws):
    slow = Slow()
    session = Agent(slow.provider()).session(ws)

    async def cancelled(prompt, *reason):
        slow.started.clear()
        stream = session.stream(prompt)
        waiting = asyncio.ensure_future(stream.wait())
        await slow.started.wait()
        assert session.cancel(*reason) is True
        return await waiting

    async def go():
        return (
            await cancelled("take your time", "stopped by the user"),
            # the reason was that cancel's alone
            await cancelled("and again"),
        )

    first, second = asyncio.run(go())
    assert (first.status, first.message) == ("cancelled", "stopped by the user")
    assert second.message == STOPPED
    run = session.runs[0]
    assert run.messages[-1].text.startswith(f"{CLOSING_NOTE} stopped by the user.")


def test_a_later_cancel_keeps_the_reason_an_earlier_one_gave(ws):
    slow = Slow()
    session = Agent(slow.provider()).session(ws)

    async def go():
        stream = session.stream("take your time")
        waiting = asyncio.ensure_future(stream.wait())
        await slow.started.wait()
        # both before the turn handles either
        assert session.cancel("stopped by the user") is True
        assert session.cancel() is True
        assert session.cancel("shutting down") is True
        return await waiting

    outcome = asyncio.run(go())
    assert outcome.message == "stopped by the user"


def test_cancel_between_tool_calls_answers_the_rest(ws):
    """A reply with two calls, cancelled after the first: the second
    gets a result saying it never ran, so the next request is valid."""
    from nontainer.conformance.corpus import ModelStep
    from nontainer.conformance.corpus import ToolCall as Call

    two = ModelStep(
        tool_calls=(
            Call(name="file_write", args={"path": "/workspace/a.txt", "content": "A"}),
            Call(name="file_write", args={"path": "/workspace/b.txt", "content": "B"}),
        )
    )
    provider = ScriptedProvider([two, says("after")])
    session = Agent(provider).session(ws)
    original = session._call

    async def then_cancel(call):
        result = await original(call)
        session.cancel()
        return result

    session._call = then_cancel
    outcome = session.say("write both")
    assert outcome.status == "cancelled"
    assert ws.files.read("/workspace/a.txt") == b"A"
    assert not ws.files.fs.exists("/workspace/b.txt")
    (run,) = session.runs
    (tool,) = [m for m in run.messages if m.role == "tool"]
    assert [(r.content == UNANSWERED, r.is_error) for r in tool.parts] == [
        (False, False),
        (True, True),
    ]
    calls = [p for p in run.messages[1].parts if isinstance(p, ToolCall)]
    assert [c.call_id for c in calls] == [r.call_id for r in tool.parts]

    session._call = original
    assert session.say("next").status == "completed"  # the history is valid


def test_cancel_from_another_thread(ws):
    slow = Slow()
    session = Agent(slow.provider()).session(ws)

    async def go():
        stream = session.stream("go")
        waiting = asyncio.ensure_future(stream.wait())
        await slow.started.wait()
        threading.Thread(target=session.cancel).start()
        return await waiting

    assert asyncio.run(go()).status == "cancelled"


@pytest.fixture
def unbounded():
    """A workspace whose code no tick limit or near timeout ends, so
    only a stop does."""
    from nontainer import PythonConfig

    store = Store(memory=True)
    ws = store.open("s", python=PythonConfig(timeout=60, tick_limit=None))
    yield ws
    ws.close()
    store.close()


def test_a_cancel_interrupts_the_tool_call_running(unbounded):
    """The call is stopped where it runs rather than waited out: code
    that would loop forever ends, and so does the turn, cancelled."""
    from nontainer.conformance.corpus import calls

    ws = unbounded
    forever = "open('started.txt', 'w').write('x')\nwhile True:\n    pass\n"
    session = agent(calls("run_python", code=forever), says("never said")).session(ws)

    async def go():
        stream = session.stream("loop")
        async for event in stream:
            if isinstance(event, ToolStarted):
                threading.Timer(0.3, session.cancel).start()
        return session.last

    started = time.monotonic()
    outcome = asyncio.run(go())
    assert outcome.status == "cancelled"
    assert time.monotonic() - started < 20  # not the 60s the code had
    assert ws.files.read("started.txt") == b"x"  # what it did is kept
    (run,) = session.runs
    (tool,) = [m for m in run.messages if m.role == "tool"]
    assert tool.parts[0].content == CUT_OFF


def test_a_call_not_yet_running_when_the_stop_came_is_stopped_too(unbounded):
    """A stop that lands before the call reaches its sandbox finds
    nothing to interrupt; ending the turn interrupts it once it does,
    rather than wait out the call."""
    from nontainer.conformance.corpus import calls

    ws = unbounded
    exec_python = ws.runtime.exec_python

    def late(*args, **kw):  # holds the workspace, not yet in the sandbox
        time.sleep(0.5)
        return exec_python(*args, **kw)

    ws.runtime.exec_python = late
    session = agent(
        calls("run_python", code="while True:\n    pass\n"), says("never said")
    ).session(ws)

    async def go():
        stream = session.stream("loop")
        async for event in stream:
            if isinstance(event, ToolStarted):
                threading.Timer(0.1, session.cancel).start()
        return session.last

    started = time.monotonic()
    assert asyncio.run(go()).status == "cancelled"
    assert time.monotonic() - started < 20  # not the 60s the code had


def test_a_turn_left_running_at_loop_shutdown_still_lands(ws):
    """Cancelled from outside (the loop closing), the turn still ends:
    stored as cancelled, its workspace free for the next turn."""
    slow = Slow()
    session = Agent(slow.provider()).session(ws)

    async def go():
        stream = session.stream("go")
        await stream.__anext__()
        await slow.started.wait()
        # return with the turn still waiting on its model call

    asyncio.run(go())
    assert ws.turns.current is None
    (run,) = session.runs
    assert run.status == "cancelled" and run.messages[-1].text.startswith(CLOSING_NOTE)


# -- failed ----------------------------------------------------------------------


def test_a_run_stopped_at_max_steps_closes_with_a_note(ws):
    from nontainer.conformance.corpus import calls

    session = agent(*[calls("terminal", command="true")] * 3, max_steps=2).session(ws)
    outcome = session.say("go")
    assert outcome.status == "failed"
    (run,) = session.runs
    assert run.messages[-1].text.startswith(CLOSING_NOTE)
    # the second reply's call was answered before the turn stopped
    assert run.messages[-2].role == "tool"


def test_a_tool_call_cut_off_mid_flight_is_ended_in_the_stream(ws):
    """Cancelled while a tool call is in flight: the stream ends that
    call too, so a UI watching it closes what it opened, and the run
    answers the call with a result saying it never returned."""
    from nontainer.conformance.corpus import calls

    session = Agent(ScriptedProvider([calls("terminal", command="true")])).session(ws)

    async def go():
        entered = asyncio.Event()

        async def stuck(call):
            entered.set()
            await asyncio.sleep(10)

        session._call = stuck
        stream = session.stream("go")
        waiting = asyncio.ensure_future(stream.wait())
        await entered.wait()
        session.cancel()
        return await waiting

    outcome = asyncio.run(go())
    assert outcome.status == "cancelled"
    ended = [e for e in outcome.events if isinstance(e, ToolEnded)]
    assert len(ended) == 1 and ended[0].is_error and ended[0].result == CUT_OFF
    assert isinstance(outcome.events[-1], RunEnded)
    (run,) = session.runs
    (tool,) = [m for m in run.messages if m.role == "tool"]
    assert [r.content for r in tool.parts] == [CUT_OFF]


def test_a_cancel_before_the_turn_has_started_stops_it_at_its_first_check(ws):
    session = agent(says("never reached")).session(ws)

    async def go():
        stream = session.stream("go")
        waiting = asyncio.ensure_future(stream.wait())
        await asyncio.sleep(0)  # the task exists, and may not have run yet
        assert session.cancel() is True
        return [e async for e in stream], await waiting

    events, outcome = asyncio.run(go())
    assert outcome.status == "cancelled"
    assert [type(e).__name__ for e in events] == ["RunStarted", "RunEnded"]
    (run,) = session.runs
    assert [m.role for m in run.messages] == ["user", "assistant"]
    assert run.messages[-1].text.startswith(CLOSING_NOTE)


def test_a_stream_whose_task_was_cancelled_before_it_ran_still_ends(ws):
    """The end of the stream comes from the task finishing, so even a
    task cancelled before its first line closes the stream."""
    session = agent(says("x")).session(ws)

    async def go():
        stream = session.stream("go")
        task = stream._start()
        task.cancel()  # straight at the task: before _run was entered
        return [e async for e in stream]

    assert asyncio.run(go()) == []
    assert not session.running and ws.turns.current is None


def test_a_cancel_during_delivery_keeps_what_the_tool_did(ws):
    """The tool ran (and wrote its file) before the turn was cancelled
    while its result was being delivered: the run keeps the tool's own
    output, so the model is not told to do it again, and the notes that
    never reached it go back to the queue."""
    from nontainer.inbox import Inbox

    reached = []

    async def go():
        stalled = asyncio.Event()

        async def slow(notes):
            reached.extend(notes)
            stalled.set()
            await asyncio.sleep(10)

        session = agent(writes("/workspace/a.txt", "A"), says("never")).session(
            ws, inbox=Inbox(on_delivered=slow)
        )
        note = session.inbox.put("also b")
        stream = session.stream("write a")
        waiting = asyncio.ensure_future(stream.wait())
        await stalled.wait()
        session.cancel()
        return session, note, await waiting

    session, note, outcome = asyncio.run(go())
    assert outcome.status == "cancelled"
    assert ws.files.read("/workspace/a.txt") == b"A"
    (ended,) = [e for e in outcome.events if isinstance(e, ToolEnded)]
    assert not ended.is_error and ended.result not in (CUT_OFF, UNANSWERED)
    (run,) = session.runs
    (tool,) = [m for m in run.messages if m.role == "tool"]
    (result,) = tool.parts
    assert result.content == ended.result and not result.is_error
    assert "also b" not in result.content
    assert session.inbox.pending() == [note] and reached == [note]
