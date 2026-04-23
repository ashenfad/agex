"""Tests for automatic chaptering trigger via _maybe_chapter."""

import pytest

from agex import Agent, clear_agent_registry, connect_state, events
from agex.agent.events import ChapterEvent
from agex.llm.dummy_client import Dummy
from tests.agex._emissions import make_response


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestAutoChapterTrigger:
    def test_chaptering_fires_when_above_trigger(self):
        """After a task whose input_tokens exceeds the trigger, _maybe_chapter
        should run the __chapter__ task and produce ChapterEvents."""
        responses = [
            # Response for the first "work" task — input_tokens above trigger
            make_response(
                thinking="Doing work.",
                code="task_success('done')",
                input_tokens=60000,
            ),
            # Response for the auto-triggered __chapter__ task
            make_response(
                thinking="Chaptering completed work.",
                code=(
                    'task_success([Chapter(start=1, end=1, name="Work phase", '
                    'message="Did the work and finished.")])'
                ),
                input_tokens=60000,
            ),
        ]
        llm = Dummy(responses=responses)
        agent = Agent(
            name="auto_ch",
            llm=llm,
            state=connect_state(type="versioned", storage="memory"),
            chaptering_trigger=50000,
        )

        @agent.task
        def do_work(msg: str) -> str:
            """Do some work."""
            pass

        result = do_work("go")
        assert result == "done"

        # The __chapter__ task should have fired automatically
        all_events = events(agent.state())
        chapter_events = [e for e in all_events if isinstance(e, ChapterEvent)]
        assert len(chapter_events) == 1
        assert chapter_events[0].name == "Work phase"
        assert chapter_events[0].message == "Did the work and finished."

    def test_chaptering_does_not_fire_below_trigger(self):
        """When input_tokens is below the trigger, no chaptering should occur."""
        responses = [
            make_response(
                thinking="Small task.",
                code="task_success('small')",
                input_tokens=30000,
            ),
        ]
        llm = Dummy(responses=responses)
        agent = Agent(
            name="no_ch",
            llm=llm,
            state=connect_state(type="versioned", storage="memory"),
            chaptering_trigger=50000,
        )

        @agent.task
        def small_task(msg: str) -> str:
            """Small task."""
            pass

        result = small_task("go")
        assert result == "small"

        all_events = events(agent.state())
        chapter_events = [e for e in all_events if isinstance(e, ChapterEvent)]
        assert len(chapter_events) == 0

    def test_chaptering_does_not_fire_without_trigger_set(self):
        """When chaptering_trigger is None, no chaptering should occur."""
        responses = [
            make_response(
                thinking="Work.",
                code="task_success('done')",
                input_tokens=100000,
            ),
        ]
        llm = Dummy(responses=responses)
        agent = Agent(
            name="no_trigger",
            llm=llm,
            state=connect_state(type="versioned", storage="memory"),
            # No chaptering_trigger set
        )

        @agent.task
        def work(msg: str) -> str:
            """Work."""
            pass

        result = work("go")
        assert result == "done"

        all_events = events(agent.state())
        chapter_events = [e for e in all_events if isinstance(e, ChapterEvent)]
        assert len(chapter_events) == 0

    def test_on_event_receives_chapter_events(self):
        """The on_event callback should receive ChapterEvents for live UI updates."""
        responses = [
            make_response(
                thinking="Doing work.",
                code="task_success('done')",
                input_tokens=60000,
            ),
            make_response(
                thinking="Chaptering.",
                code=(
                    'task_success([Chapter(start=1, end=1, name="Phase 1", '
                    'message="Completed phase 1.")])'
                ),
                input_tokens=60000,
            ),
        ]
        llm = Dummy(responses=responses)
        agent = Agent(
            name="live_ch",
            llm=llm,
            state=connect_state(type="versioned", storage="memory"),
            chaptering_trigger=50000,
        )

        @agent.task
        def do_work(msg: str) -> str:
            """Do some work."""
            pass

        received = []
        result = do_work("go", on_event=received.append)
        assert result == "done"

        chapter_events = [e for e in received if isinstance(e, ChapterEvent)]
        assert len(chapter_events) == 1
        assert chapter_events[0].name == "Phase 1"
