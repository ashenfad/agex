"""Tests for ChapterEvent (creation, rendering, token counting, nesting)."""

import pytest

from agex import clear_agent_registry
from agex.agent.events import (
    ChapterEvent,
)


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestChapterEventCreation:
    def test_basic_creation(self):
        ch = ChapterEvent(
            agent_name="test",
            name="Data exploration",
            message="Found 3 tables",
        )
        assert ch.name == "Data exploration"
        assert ch.message == "Found 3 tables"
        assert ch.event_refs == []

    def test_creation_with_refs(self):
        ch = ChapterEvent(
            agent_name="test",
            name="Work phase",
            message="Did some work",
            event_refs=["_event_1_", "_event_2_"],
        )
        assert len(ch.event_refs) == 2

    def test_nested_chapters(self):
        outer = ChapterEvent(
            agent_name="t",
            name="Outer",
            message="Outer work",
            event_refs=["ref_to_inner"],
        )
        assert len(outer.event_refs) == 1


class TestChapterEventTokenCounting:
    def test_tokens_computed(self):
        ch = ChapterEvent(agent_name="t", name="Test", message="A summary")
        assert ch.full_detail_tokens > 0
        assert ch.low_detail_tokens > 0
        assert ch.full_detail_tokens == ch.low_detail_tokens

    def test_tokens_based_on_summary_not_events(self):
        """Token count should reflect the summary, not embedded events."""
        ch = ChapterEvent(
            agent_name="t",
            name="Test",
            message="Short summary",
            event_refs=["ref1", "ref2", "ref3"],
        )
        # Tokens should be small since they're based on the summary
        assert ch.full_detail_tokens < 50


class TestChapterEventRendering:
    def test_str(self):
        ch = ChapterEvent(
            agent_name="t",
            name="Exploration",
            message="Found data",
            event_refs=["ref1"],
        )
        s = str(ch)
        assert 'Chapter: "Exploration"' in s
        assert "(1 events)" in s

    def test_repr_markdown(self):
        ch = ChapterEvent(
            agent_name="t",
            name="Exploration",
            message="Found 3 tables",
            event_refs=[],
        )
        md = ch._repr_markdown_()
        assert "📖" in md
        assert "Exploration" in md
        assert "Found 3 tables" in md

    def test_repr_html(self):
        ch = ChapterEvent(
            agent_name="t",
            name="Exploration",
            message="Found data",
        )
        html = ch._repr_html_()
        assert "📖" in html
        assert "Exploration" in html
        assert "ChapterEvent" in html
