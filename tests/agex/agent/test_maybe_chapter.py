"""Direct unit tests for ``Agent._maybe_chapter``.

These tests bypass the sandbox by stubbing ``agent._chapter_task`` —
they pre-populate the parent's event log with synthetic events and
verify the boundary-index → ChapterEvent wiring without spinning up
a full task loop.
"""

import asyncio
from typing import Any

import pytest

from agex import Agent, clear_agent_registry, connect_state, events
from agex.agent.chapter import CHAPTER_TASK, Chapter
from agex.agent.events import ChapterEvent, TaskStartEvent
from agex.llm.dummy_client import Dummy
from agex.state.log import add_event_to_log
from tests.agex._emissions import make_action_event


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


def _make_agent(name: str = "ch") -> Agent:
    """Agent with chaptering enabled, dummy LLM (no actual calls)."""
    return Agent(
        name=name,
        llm=Dummy(responses=[]),
        state=connect_state(type="versioned", storage="memory"),
        chaptering_trigger=50000,
    )


def _populate_log(state, *events_):
    for ev in events_:
        add_event_to_log(state, ev)


def _success(result="ok", agent="ch"):
    from agex.agent.events import SuccessEvent

    return SuccessEvent(agent_name=agent, result=result)


def _ts(name: str, agent: str = "ch", message: str = "") -> TaskStartEvent:
    return TaskStartEvent(
        agent_name=agent,
        task_name=name,
        inputs={"message": message or f"do {name}"},
        message=message or f"do {name}",
    )


class FakeChapterTask:
    """Records calls and returns a configured chapters list.

    Mirrors the agent.task() wrapper interface: it's awaitable and
    accepts ``event_index``, ``session``, ``on_event``, ``on_token``.
    """

    def __init__(self, returns: list[Chapter] | Exception | None = None):
        self.returns = returns or []
        self.called_with: dict[str, Any] | None = None
        self.call_count = 0

    async def __call__(self, *, event_index, session, on_event, on_token):
        self.call_count += 1
        self.called_with = {
            "event_index": event_index,
            "session": session,
        }
        if isinstance(self.returns, Exception):
            raise self.returns
        return list(self.returns)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestMaybeChapter:
    def test_no_op_when_chapter_task_disabled(self):
        agent = Agent(
            name="no_ch",
            llm=Dummy(responses=[]),
            state=connect_state(type="versioned", storage="memory"),
            # No chaptering_trigger configured.
        )
        state = agent.state()
        _populate_log(
            state,
            _ts("t1"),
            make_action_event(
                agent_name="ch", thinking="t", code="x", input_tokens=99999
            ),
            _success(),
        )
        # _chapter_task is None → early return, log unchanged.
        before = list(events(state))
        _run(agent._maybe_chapter(state, "default", on_event=None, on_token=None))
        after = list(events(state))
        assert [type(e) for e in before] == [type(e) for e in after]

    def test_no_op_below_trigger(self):
        agent = _make_agent()
        agent._chapter_task = FakeChapterTask()
        state = agent.state()
        _populate_log(
            state,
            _ts("t1"),
            make_action_event(
                agent_name="ch", thinking="t", code="x", input_tokens=10000
            ),
            _success(),
        )
        _run(agent._maybe_chapter(state, "default", None, None))
        assert agent._chapter_task.call_count == 0

    def test_no_op_when_no_completable_boundary(self):
        """Single in-progress task — even though token threshold is
        exceeded, ``has_completable_boundary`` returns False so the
        chapter task is NOT invoked.  No bookkeeping events get added."""
        agent = _make_agent()
        agent._chapter_task = FakeChapterTask(
            returns=[Chapter(start=1, end=1, name="X", message="never called")]
        )
        state = agent.state()
        _populate_log(
            state,
            _ts("running"),  # in-progress (no terminator)
            make_action_event(
                agent_name="ch", thinking="t", code="x", input_tokens=99999
            ),
        )
        _run(agent._maybe_chapter(state, "default", None, None))
        assert agent._chapter_task.call_count == 0
        # No ChapterEvent in the log.
        assert all(not isinstance(e, ChapterEvent) for e in events(state))

    def test_invokes_chapter_task_with_boundary_index(self):
        agent = _make_agent()
        agent._chapter_task = FakeChapterTask(
            returns=[Chapter(start=1, end=1, name="Phase 1", message="completed t1")]
        )
        state = agent.state()
        _populate_log(
            state,
            _ts("t1", message="do t1"),
            make_action_event(agent_name="ch", thinking="t", code="x"),
            _success(result="r1"),
            # Triggering action: above threshold, opens an in-progress
            # task that's NOT eligible for folding.
            _ts("running"),
            make_action_event(
                agent_name="ch", thinking="t", code="x", input_tokens=99999
            ),
        )
        _run(agent._maybe_chapter(state, "default", None, None))

        assert agent._chapter_task.call_count == 1
        # The index handed to the chapter task contains both boundaries
        # (t1 with success, running task as in-progress).
        text = agent._chapter_task.called_with["event_index"]
        assert "[1] task " in text
        assert "t1" in text
        assert "→ success" in text
        assert "[2] task " in text
        assert "(in progress)" in text
        # CHAPTER_TASK is filtered from the index — never visible to itself.
        assert CHAPTER_TASK not in text

    def test_chapter_event_replaces_log_range_correctly(self):
        agent = _make_agent()
        agent._chapter_task = FakeChapterTask(
            returns=[Chapter(start=1, end=1, name="Phase 1", message="t1 done")]
        )
        state = agent.state()
        _populate_log(
            state,
            _ts("t1"),  # 0
            make_action_event(agent_name="ch", thinking="t", code="x"),  # 1
            _success(result="r1"),  # 2
            _ts("running"),  # 3 — boundary 2, in-progress
            make_action_event(
                agent_name="ch", thinking="t", code="x", input_tokens=99999
            ),  # 4
        )
        _run(agent._maybe_chapter(state, "default", None, None))

        all_events = list(events(state))
        # First three events (the t1 boundary's range) get replaced
        # by a single ChapterEvent; the running task survives unchanged.
        assert isinstance(all_events[0], ChapterEvent)
        assert all_events[0].name == "Phase 1"
        assert isinstance(all_events[1], TaskStartEvent)
        assert all_events[1].task_name == "running"
        # The chapter's event_refs covers exactly the t1 boundary's
        # log range (0..3 → 3 refs).
        assert len(all_events[0].event_refs) == 3

    def test_chapter_task_failure_no_log_changes(self):
        """A raising chapter task is logged but does not corrupt the
        parent's event log."""
        agent = _make_agent()
        agent._chapter_task = FakeChapterTask(returns=RuntimeError("boom"))
        state = agent.state()
        _populate_log(
            state,
            _ts("t1"),
            _success(),
            _ts("running"),
            make_action_event(
                agent_name="ch", thinking="t", code="x", input_tokens=99999
            ),
        )
        n_before = len(list(events(state)))
        _run(agent._maybe_chapter(state, "default", None, None))
        n_after = len(list(events(state)))
        # Failure path doesn't add or remove events on this side
        # (the chapter task's own bookkeeping would have, but our fake
        # raises before any of that runs).
        assert n_after == n_before

    def test_emits_chapter_events_to_on_event(self):
        received = []

        def on_event(ev):
            received.append(ev)

        agent = _make_agent()
        agent._chapter_task = FakeChapterTask(
            returns=[Chapter(start=1, end=1, name="P1", message="summary")]
        )
        state = agent.state()
        _populate_log(
            state,
            _ts("t1"),
            _success(),
            _ts("running"),
            make_action_event(
                agent_name="ch", thinking="t", code="x", input_tokens=99999
            ),
        )
        _run(agent._maybe_chapter(state, "default", on_event, None))
        # The on_event callback received the ChapterEvent for live UI updates.
        chapter_events = [e for e in received if isinstance(e, ChapterEvent)]
        assert len(chapter_events) == 1
        assert chapter_events[0].name == "P1"

    def test_nested_chaptering_chapter_event_as_boundary(self):
        """Folding a range that includes a prior ChapterEvent produces
        an outer chapter whose event_refs includes the inner chapter's
        storage key (nested chaptering)."""
        agent = _make_agent()
        agent._chapter_task = FakeChapterTask(
            returns=[
                # Fold both the prior chapter (boundary 1) and the
                # completed t2 (boundary 2) into one outer chapter.
                Chapter(start=1, end=2, name="Outer", message="early phases"),
            ]
        )
        state = agent.state()
        # First fabricate a prior ChapterEvent in the log.
        prior_chapter = ChapterEvent(agent_name="ch", name="inner", message="phase 1")
        _populate_log(
            state,
            prior_chapter,  # 0 — boundary 1
            _ts("t2"),  # 1 — boundary 2 starts
            _success(result="r2"),  # 2
            _ts("running"),  # 3 — boundary 3, in-progress
            make_action_event(
                agent_name="ch", thinking="t", code="x", input_tokens=99999
            ),  # 4
        )
        _run(agent._maybe_chapter(state, "default", None, None))

        all_events = list(events(state))
        # The outer chapter replaces events[0:3] (prior chapter +
        # t2 boundary's range).
        assert isinstance(all_events[0], ChapterEvent)
        assert all_events[0].name == "Outer"
        # event_refs covers 3 underlying log entries — the inner
        # chapter's key, plus the t2 TaskStart and Success.
        assert len(all_events[0].event_refs) == 3
        # The inner chapter is still resolvable — chapters_vfs uses
        # this to render the nested ``/chapters/outer/chapters/inner/``
        # tree.
        nested = all_events[0].resolve_events(state)
        nested_chapters = [e for e in nested if isinstance(e, ChapterEvent)]
        assert len(nested_chapters) == 1
        assert nested_chapters[0].name == "inner"
