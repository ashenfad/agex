"""Tests for the public chapter API surface in ``agex/agent/chapter.py``.

The boundary-index helpers (``build_chapter_scope_filter``,
``build_boundary_index``, ``has_completable_boundary``) are exercised
in ``test_chapter_boundary.py``.
"""

import pytest

from agex import clear_agent_registry
from agex.agent.chapter import (
    CHAPTER_TASK_PRIMER,
    Chapter,
    should_trigger_chaptering,
)
from agex.agent.events import (
    OutputEvent,
    TaskStartEvent,
)
from tests.agex._emissions import make_action_event


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

    def test_primer_defaults_to_folding(self):
        """Primer must frame folding as the default action — the
        chapter task only runs when context is over budget, so an
        agent that learns 'empty list is fine' undermines the urgency
        and pollutes the log with no-op bookkeeping each time."""
        # Action-by-default framing.
        assert "default to folding" in CHAPTER_TASK_PRIMER
        # Empty list is reframed as a last resort.
        assert "last resort" in CHAPTER_TASK_PRIMER

    def test_primer_mentions_nested_chaptering(self):
        """A prior chapter entry can be folded into a deeper outer
        chapter — the agent needs to know this is normal, not an error."""
        assert "nested chaptering" in CHAPTER_TASK_PRIMER

    def test_primer_uses_python_syntax(self):
        """``Chapter(...)`` constructor + ``task_success([...])`` —
        not the agex-ts object-literal / ``taskSuccess`` shape."""
        assert "Chapter(" in CHAPTER_TASK_PRIMER
        assert "task_success(" in CHAPTER_TASK_PRIMER
        # Sanity: no leaked TS syntax.
        assert "taskSuccess" not in CHAPTER_TASK_PRIMER


class TestShouldTriggerChaptering:
    def test_none_threshold_returns_false(self):
        events = [
            make_action_event(
                agent_name="t", thinking="t", code="x", input_tokens=100000
            )
        ]
        assert should_trigger_chaptering(events, None) is False

    def test_below_threshold(self):
        events = [
            make_action_event(
                agent_name="t", thinking="t", code="x", input_tokens=50000
            )
        ]
        assert should_trigger_chaptering(events, 100000) is False

    def test_above_threshold(self):
        events = [
            make_action_event(
                agent_name="t", thinking="t", code="x", input_tokens=150000
            )
        ]
        assert should_trigger_chaptering(events, 100000) is True

    def test_uses_most_recent_action_event(self):
        events = [
            make_action_event(
                agent_name="t", thinking="t", code="x", input_tokens=150000
            ),
            OutputEvent(agent_name="t", parts=[]),
            make_action_event(
                agent_name="t", thinking="t", code="x", input_tokens=50000
            ),
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
            make_action_event(agent_name="t", thinking="t", code="x", input_tokens=None)
        ]
        assert should_trigger_chaptering(events, 100000) is False

    def test_empty_events(self):
        assert should_trigger_chaptering([], 100000) is False
