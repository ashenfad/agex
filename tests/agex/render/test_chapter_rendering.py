"""Tests for ChapterEvent rendering in markdown and XML formats."""

from agex.agent.events import (
    ActionEvent,
    ChapterEvent,
    FailEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.render.events import render_events_as_markdown
from agex.render.primitives import render_chapter
from agex.render.xml import render_events_as_xml


class TestRenderChapter:
    def test_basic_render(self):
        text, tokens = render_chapter("Data exploration", "Found 3 tables")
        assert '📖 Chapter: "Data exploration"' in text
        assert "Found 3 tables" in text
        assert tokens > 0

    def test_empty_message(self):
        text, tokens = render_chapter("Empty", "")
        assert '📖 Chapter: "Empty"' in text
        assert tokens > 0


class TestChapterEventMarkdownRendering:
    def test_chapter_rendered_as_user_message(self):
        events = [
            ChapterEvent(
                agent_name="t",
                name="Exploration",
                message="Found stuff",
            )
        ]
        messages = render_events_as_markdown(events)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Exploration" in messages[0]["content"]
        assert "Found stuff" in messages[0]["content"]

    def test_chapter_in_event_sequence(self):
        events = [
            TaskStartEvent(agent_name="t", task_name="task", inputs={}, message="Go"),
            ChapterEvent(
                agent_name="t",
                name="Early work",
                message="Did some setup",
            ),
            ActionEvent(
                agent_name="t", thinking="think", code="x = 1", title="Next step"
            ),
            SuccessEvent(agent_name="t", result=42),
        ]
        messages = render_events_as_markdown(events)
        # Consecutive same-role messages are collapsed:
        # user(TaskStart) + user(Chapter) → 1 user, asst(Action) + asst(Success) → 1 asst
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert "Go" in messages[0]["content"]
        assert "Early work" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"

    def test_multiple_chapters(self):
        events = [
            ChapterEvent(agent_name="t", name="Phase 1", message="Setup"),
            ChapterEvent(agent_name="t", name="Phase 2", message="Analysis"),
        ]
        messages = render_events_as_markdown(events)
        # Two consecutive user messages collapsed into one
        assert len(messages) == 1
        assert "Phase 1" in messages[0]["content"]
        assert "Phase 2" in messages[0]["content"]


class TestChapterEventXMLRendering:
    def test_chapter_rendered_as_user_message(self):
        events = [
            ChapterEvent(
                agent_name="t",
                name="Exploration",
                message="Found stuff",
            )
        ]
        messages = render_events_as_xml(events)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Exploration" in messages[0]["content"]

    def test_chapter_in_xml_sequence(self):
        events = [
            ChapterEvent(agent_name="t", name="Setup", message="Done"),
            ActionEvent(agent_name="t", thinking="think", code="x = 1", title="Act"),
            FailEvent(agent_name="t", message="oops"),
        ]
        messages = render_events_as_xml(events)
        # Chapter(user), then Action+Fail(assistant) collapsed
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "oops" in messages[1]["content"]
