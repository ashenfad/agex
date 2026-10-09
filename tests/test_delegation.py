"""Delegation: a session asks through its ``sessions`` tool, and each
delegate is an agex session of the same agent on a fork of its world
(``agex.delegation.Runner``)."""

import asyncio
import time

import pytest
from nontainer import Profile, Store
from nontainer.conformance.corpus import ModelStep, ToolCall, asks, fails, says, writes
from nontainer.workspace import Workspace

from agex import Agent
from agex.agent import Session
from agex.delegation import Runner
from agex.providers.scripted import ScriptedProvider


@pytest.fixture
def store():
    st = Store(memory=True)
    yield st
    st.close()


def routed(**scripts):
    """One scripted model for every session: a request goes to the
    script whose key opens the session's first message, so a parent
    and its delegates each follow their own."""
    queues = {key: list(steps) for key, steps in scripts.items()}

    def next_step(sent):
        first = sent[1] if len(sent) > 1 else ""
        for key, queue in queues.items():
            if first.startswith(key):
                assert queue, f"{key}'s script ran out"
                return queue.pop(0)
        raise AssertionError(f"no script for a session opening {first[:60]!r}")

    return Agent(ScriptedProvider(next_step))


def ask(name, task, **more):
    args = {"action": "ask", "name": name, "task": task, **more}
    return ModelStep(tool_calls=(ToolCall(name="sessions", args=args),))


def test_a_delegate_does_the_work_and_its_answer_comes_back(store):
    agent = routed(
        LEAD=[ask("scout", "SCOUT write s.txt", wait=True), says("the scout wrote it")],
        SCOUT=[writes("/workspace/s.txt", "S"), says("wrote s.txt")],
    )
    ws = store.open("lead")
    chat = agent.session(ws, sessions=True)
    try:
        outcome = chat.say("LEAD get it written")
        assert (outcome.status, outcome.text) == ("completed", "the scout wrote it")
        [job] = chat.sessions.list()
        assert (job.name, job.status) == ("lead.scout", "answered")
        answer = chat.sessions.result("lead.scout")
        assert answer.text == "wrote s.txt"
        assert answer.changed  # the delegate's fork holds its work
        # the caller's world is its own: the work is on the delegate's branch
        assert not ws.files.exists("/workspace/s.txt")
        scout = store.open("lead.scout")
        try:
            assert scout.files.read("/workspace/s.txt") == b"S"
        finally:
            scout.close()
        # the answer was the result the lead's model read
        [run] = chat.runs
        assert "wrote s.txt" in str(run.messages[2].parts[0])
    finally:
        chat.close()
        ws.close()


def test_a_delegate_hears_from_its_own_delegate_before_it_answers(store):
    """The scout asks a sub-delegate and ends its turn to wait; the
    sub's answer wakes it, and the reply it gives then is its answer."""
    agent = routed(
        LEAD=[ask("scout", "SCOUT find it", wait=True), says("done")],
        SCOUT=[
            asks("sub", "SUB look in the attic"),
            says("waiting on sub"),
            says("sub found it in the attic"),
        ],
        SUB=[says("it is in the attic")],
    )
    ws = store.open("lead")
    chat = agent.session(ws, sessions=True)
    try:
        chat.say("LEAD find it")
        answer = chat.sessions.result("lead.scout")
        assert answer.text == "sub found it in the attic"
        assert store.exists("lead.scout.sub")
    finally:
        chat.close()
        ws.close()


def test_a_delegate_whose_turn_fails_answers_failed(store):
    agent = routed(
        LEAD=[ask("scout", "SCOUT try", wait=True), says("it failed")],
        SCOUT=[fails("error")],
    )
    ws = store.open("lead")
    chat = agent.session(ws, sessions=True)
    try:
        chat.say("LEAD try it")
        answer = chat.sessions.result("lead.scout")
        assert answer.status == "failed"
        assert "ended failed" in answer.text
    finally:
        chat.close()
        ws.close()


def test_a_session_made_in_a_loop_runs_its_delegates_there(store):
    agent = routed(
        LEAD=[ask("scout", "SCOUT say hi", wait=True), says("it said hi")],
        SCOUT=[says("hi")],
    )

    async def main():
        ws = store.open("lead")
        chat = agent.session(ws, sessions=True)
        try:
            outcome = await chat.asay("LEAD greet")
            assert outcome.text == "it said hi"
            assert chat.sessions.result("lead.scout").text == "hi"
        finally:
            await chat.aclose()
            ws.close()

    asyncio.run(main())


def test_an_answer_that_lands_between_turns_wakes_the_next(store):
    """Asked without waiting, the answer lands once the lead's turn has
    ended; a woken turn opens with it."""
    agent = routed(
        LEAD=[ask("scout", "SCOUT note it"), says("asked"), says("the scout noted it")],
        SCOUT=[says("noted")],
    )
    ws = store.open("lead")
    chat = agent.session(ws, sessions=True)
    try:
        chat.say("LEAD have it noted")
        assert chat.sessions.wait(timeout=10) == ["lead.scout"]
        outcome = chat.wake()
        assert outcome.text == "the scout noted it"
        opening = chat.runs[-1].messages[0].text
        assert "noted" in opening and "scout" in opening
        # delivered once: nothing is left waiting
        assert chat.sessions.outstanding() == []
    finally:
        chat.close()
        ws.close()


def test_a_delegate_holds_its_parents_profile(store):
    profile = Profile(variables={"WHO": "the lead"})
    ws = store.open("lead", profile=profile)
    try:
        runner = Runner(Agent(ScriptedProvider([])), ws)
        assert runner.profile == Profile.of(ws)
        assert runner.profile.variables == {"WHO": "the lead"}
    finally:
        ws.close()


def test_a_workspace_from_no_store_cannot_delegate(tmp_path):
    from nontainer.providers import KvgitProvider

    ws = Workspace(KvgitProvider.open(tmp_path / "kv", session="solo"))
    try:
        with pytest.raises(ValueError, match="opened from a Store"):
            Agent(ScriptedProvider([])).session(ws, sessions=True)
    finally:
        ws.close()


def test_cancelling_a_delegate_stops_its_turn(store):
    agent = routed(
        LEAD=[ask("scout", "SCOUT take your time"), says("asked")],
        SCOUT=[
            ModelStep(
                tool_calls=(
                    ToolCall(
                        name="run_python", args={"code": "import time; time.sleep(1)"}
                    ),
                )
            ),
            says("unreachable"),
        ],
    )
    ws = store.open("lead")
    chat = agent.session(ws, sessions=True)
    try:
        chat.say("LEAD start it")
        until = time.monotonic() + 10
        while not store.exists("lead.scout") and time.monotonic() < until:
            time.sleep(0.01)
        time.sleep(0.2)  # into the scout's run_python call
        job = chat.sessions.cancel("lead.scout")
        assert job.status == "cancelled"
    finally:
        chat.close()  # waits for the scout's run to wind down
        ws.close()
    scout = store.open("lead.scout")
    try:
        [run] = Session(agent, scout).runs
        assert run.status == "cancelled"
    finally:
        scout.close()
