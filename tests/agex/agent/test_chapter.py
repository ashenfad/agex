"""Tests for chapter support (agex/agent/chapter.py)."""

import pytest

from agex import clear_agent_registry
from agex.agent.chapter import (
    CHAPTER_TASK_PRIMER,
    Chapter,
    build_numbered_event_index,
    should_trigger_chaptering,
)
from agex.agent.events import (
    ActionEvent,
    CancelledEvent,
    ChapterEvent,
    ClarifyEvent,
    ErrorEvent,
    FailEvent,
    FileEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.eval.objects import PrintAction


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


class TestBuildNumberedEventIndex:
    def test_empty_events(self):
        assert build_numbered_event_index([]) == ""

    def test_task_start_event(self):
        events = [
            TaskStartEvent(
                agent_name="t", task_name="analyze", inputs={}, message="msg"
            )
        ]
        result = build_numbered_event_index(events)
        assert '[1] Task: "analyze"' in result

    def test_action_event_with_code(self):
        events = [
            ActionEvent(
                agent_name="t",
                thinking="think",
                code="x = 1\ny = 2",
                title="Compute values",
            )
        ]
        result = build_numbered_event_index(events)
        assert "[1] Action: Compute values (2 lines)" in result

    def test_action_event_with_terminal(self):
        events = [
            ActionEvent(
                agent_name="t",
                thinking="think",
                code=None,
                terminal="ls -la",
                title="List files",
            )
        ]
        result = build_numbered_event_index(events)
        assert "[1] Action: List files (terminal)" in result

    def test_action_event_no_title(self):
        events = [ActionEvent(agent_name="t", thinking="think", code="x = 1")]
        result = build_numbered_event_index(events)
        assert "[1] Action: untitled (1 lines)" in result

    def test_output_event_empty(self):
        events = [OutputEvent(agent_name="t", parts=[])]
        result = build_numbered_event_index(events)
        assert "[1] Output: (empty)" in result

    def test_output_event_with_print(self):
        pa = PrintAction(["Hello world"])
        events = [OutputEvent(agent_name="t", parts=[pa])]
        result = build_numbered_event_index(events)
        assert "[1] Output: Hello world" in result

    def test_success_event(self):
        events = [SuccessEvent(agent_name="t", result=42)]
        result = build_numbered_event_index(events)
        assert "[1] Success: 42" in result

    def test_fail_event(self):
        events = [FailEvent(agent_name="t", message="Something broke")]
        result = build_numbered_event_index(events)
        assert "[1] Fail: Something broke" in result

    def test_clarify_event(self):
        events = [ClarifyEvent(agent_name="t", message="Need more info")]
        result = build_numbered_event_index(events)
        assert "[1] Clarify: Need more info" in result

    def test_cancelled_event(self):
        events = [
            CancelledEvent(agent_name="t", task_name="analyze", iterations_completed=5)
        ]
        result = build_numbered_event_index(events)
        assert "[1] Cancelled: analyze" in result

    def test_chapter_event(self):
        events = [
            ChapterEvent(
                agent_name="t",
                name="Data exploration",
                message="Found 3 tables with nulls",
            )
        ]
        result = build_numbered_event_index(events)
        assert '[1] Chapter: "Data exploration"' in result

    def test_file_event(self):
        events = [
            FileEvent(
                agent_name="t",
                file_source="agent",
                added=["a.py", "b.py"],
                modified=["c.py"],
                removed=[],
            )
        ]
        result = build_numbered_event_index(events)
        assert "[1] Files: +2 ~1" in result

    def test_pre_filtered_input(self):
        """build_numbered_event_index expects pre-filtered input (no ErrorEvents)."""
        all_events = [
            ActionEvent(agent_name="t", thinking="think", code="x = 1"),
            ErrorEvent(agent_name="t", error=RuntimeError("boom")),
            SuccessEvent(agent_name="t", result=42),
        ]
        # Caller is responsible for filtering
        visible = [e for e in all_events if not isinstance(e, ErrorEvent)]
        result = build_numbered_event_index(visible)
        lines = result.strip().split("\n")
        # Sequential numbering with no gaps
        assert len(lines) == 2
        assert "[1]" in lines[0]
        assert "[2]" in lines[1]

    def test_numbering_is_positional(self):
        events = [
            TaskStartEvent(agent_name="t", task_name="task1", inputs={}, message="msg"),
            ActionEvent(agent_name="t", thinking="t", code="x = 1", title="Step 1"),
            OutputEvent(agent_name="t", parts=[]),
            SuccessEvent(agent_name="t", result="done"),
        ]
        result = build_numbered_event_index(events)
        lines = result.strip().split("\n")
        assert len(lines) == 4
        assert lines[0].startswith("[1]")
        assert lines[1].startswith("[2]")
        assert lines[2].startswith("[3]")
        assert lines[3].startswith("[4]")

    def test_long_text_truncated(self):
        long_result = "x" * 200
        events = [SuccessEvent(agent_name="t", result=long_result)]
        result = build_numbered_event_index(events)
        assert "..." in result
        assert len(result) < 200


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
