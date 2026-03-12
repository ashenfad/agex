"""Tests for chapter support (agex/agent/chapter.py)."""

import pytest

from agex import clear_agent_registry
from agex.agent.chapter import (
    CHAPTER_TASK,
    CHAPTER_TASK_PRIMER,
    Chapter,
    build_numbered_task_index,
    prepare_tasks_for_chaptering,
    should_trigger_chaptering,
)
from agex.agent.events import (
    ActionEvent,
    ErrorEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestChapterDataclass:
    def test_basic_creation(self):
        ch = Chapter(start=1, end=3, name="Exploration", message="Found data")
        assert ch.start == 1
        assert ch.end == 3
        assert ch.name == "Exploration"
        assert ch.message == "Found data"

    def test_primer_exists(self):
        assert isinstance(CHAPTER_TASK_PRIMER, str)
        assert len(CHAPTER_TASK_PRIMER) > 50
        assert "Chapter" in CHAPTER_TASK_PRIMER
        assert "task_success" in CHAPTER_TASK_PRIMER


class TestPrepareTasksForChaptering:
    def test_empty_events(self):
        tasks, ranges = prepare_tasks_for_chaptering([])
        assert tasks == []
        assert ranges == []

    def test_single_complete_task(self):
        events = [
            TaskStartEvent(
                agent_name="t",
                task_name="analyze",
                inputs={"message": "do it"},
                message="do it",
            ),
            ActionEvent(agent_name="t", thinking="t", code="x = 1"),
            SuccessEvent(agent_name="t", result="done"),
        ]
        tasks, ranges = prepare_tasks_for_chaptering(events)
        assert len(tasks) == 1
        assert tasks[0].name == "analyze"
        assert tasks[0].complete is True
        assert "done" in tasks[0].outcome
        assert ranges == [(0, 3)]

    def test_multiple_tasks(self):
        events = [
            TaskStartEvent(agent_name="t", task_name="task1", inputs={}, message=""),
            ActionEvent(agent_name="t", thinking="t", code="x"),
            SuccessEvent(agent_name="t", result="r1"),
            TaskStartEvent(agent_name="t", task_name="task2", inputs={}, message=""),
            ActionEvent(agent_name="t", thinking="t", code="y"),
            SuccessEvent(agent_name="t", result="r2"),
        ]
        tasks, ranges = prepare_tasks_for_chaptering(events)
        assert len(tasks) == 2
        assert tasks[0].name == "task1"
        assert tasks[1].name == "task2"
        assert ranges == [(0, 3), (3, 6)]

    def test_incomplete_task(self):
        events = [
            TaskStartEvent(agent_name="t", task_name="wip", inputs={}, message=""),
            ActionEvent(agent_name="t", thinking="t", code="x"),
        ]
        tasks, ranges = prepare_tasks_for_chaptering(events)
        assert len(tasks) == 1
        assert tasks[0].complete is False
        assert ranges == [(0, 2)]

    def test_error_events_skipped(self):
        events = [
            TaskStartEvent(agent_name="t", task_name="t1", inputs={}, message=""),
            ErrorEvent(agent_name="t", error=RuntimeError("boom")),
            SuccessEvent(agent_name="t", result="ok"),
        ]
        tasks, ranges = prepare_tasks_for_chaptering(events)
        assert len(tasks) == 1
        # Range covers all events including errors
        assert ranges == [(0, 3)]

    def test_failed_task(self):
        events = [
            TaskStartEvent(agent_name="t", task_name="t1", inputs={}, message=""),
            FailEvent(agent_name="t", message="Something broke"),
        ]
        tasks, ranges = prepare_tasks_for_chaptering(events)
        assert tasks[0].complete is True
        assert "Failed" in tasks[0].outcome

    def test_chapter_task_included(self):
        """Prior __chapter__ tasks should appear in the index."""
        events = [
            TaskStartEvent(agent_name="t", task_name="t1", inputs={}, message=""),
            SuccessEvent(agent_name="t", result="r1"),
            TaskStartEvent(
                agent_name="t", task_name=CHAPTER_TASK, inputs={}, message=""
            ),
            ActionEvent(agent_name="t", thinking="t", code="x"),
            SuccessEvent(agent_name="t", result=[]),
            TaskStartEvent(agent_name="t", task_name="t2", inputs={}, message=""),
            SuccessEvent(agent_name="t", result="r2"),
        ]
        tasks, ranges = prepare_tasks_for_chaptering(events)
        assert len(tasks) == 3
        assert tasks[0].name == "t1"
        assert tasks[1].name == CHAPTER_TASK
        assert tasks[2].name == "t2"


class TestBuildNumberedTaskIndex:
    def test_empty(self):
        assert build_numbered_task_index([]) == ""

    def test_complete_task(self):
        tasks, _ = prepare_tasks_for_chaptering(
            [
                TaskStartEvent(
                    agent_name="t",
                    task_name="analyze",
                    inputs={"message": "check data"},
                    message="check data",
                ),
                SuccessEvent(agent_name="t", result="Found 3 tables"),
            ]
        )
        result = build_numbered_task_index(tasks)
        assert "[1]" in result
        assert '"analyze"' in result
        assert "Found 3 tables" in result

    def test_incomplete_task(self):
        tasks, _ = prepare_tasks_for_chaptering(
            [
                TaskStartEvent(agent_name="t", task_name="wip", inputs={}, message=""),
                ActionEvent(agent_name="t", thinking="t", code="x"),
            ]
        )
        result = build_numbered_task_index(tasks)
        assert "(in progress)" in result

    def test_numbering(self):
        tasks, _ = prepare_tasks_for_chaptering(
            [
                TaskStartEvent(agent_name="t", task_name="t1", inputs={}, message=""),
                SuccessEvent(agent_name="t", result="r1"),
                TaskStartEvent(agent_name="t", task_name="t2", inputs={}, message=""),
                SuccessEvent(agent_name="t", result="r2"),
            ]
        )
        result = build_numbered_task_index(tasks)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("[1]")
        assert lines[1].startswith("[2]")


class TestShouldTriggerChaptering:
    def test_none_threshold_returns_false(self):
        events = [
            ActionEvent(agent_name="t", thinking="t", code="x", input_tokens=100000)
        ]
        assert should_trigger_chaptering(events, None) is False

    def test_below_threshold(self):
        events = [
            ActionEvent(agent_name="t", thinking="t", code="x", input_tokens=50000)
        ]
        assert should_trigger_chaptering(events, 100000) is False

    def test_above_threshold(self):
        events = [
            ActionEvent(agent_name="t", thinking="t", code="x", input_tokens=150000)
        ]
        assert should_trigger_chaptering(events, 100000) is True

    def test_uses_most_recent_action_event(self):
        events = [
            ActionEvent(agent_name="t", thinking="t", code="x", input_tokens=150000),
            OutputEvent(agent_name="t", parts=[]),
            ActionEvent(agent_name="t", thinking="t", code="x", input_tokens=50000),
        ]
        # Most recent ActionEvent has 50k tokens, below 100k threshold
        assert should_trigger_chaptering(events, 100000) is False

    def test_no_action_events(self):
        events = [
            TaskStartEvent(agent_name="t", task_name="task", inputs={}, message="msg"),
            OutputEvent(agent_name="t", parts=[]),
        ]
        assert should_trigger_chaptering(events, 100000) is False

    def test_action_event_without_input_tokens(self):
        events = [
            ActionEvent(agent_name="t", thinking="t", code="x", input_tokens=None)
        ]
        assert should_trigger_chaptering(events, 100000) is False

    def test_empty_events(self):
        assert should_trigger_chaptering([], 100000) is False
