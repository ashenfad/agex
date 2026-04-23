"""Tests for ChapterEvent rendering in markdown and XML formats."""

from agex.agent.events import (
    ChapterEvent,
    FailEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.llm.formats.xml import render_events_as_xml
from agex.render.events import render_events_as_markdown
from agex.render.primitives import render_chapter
from tests.agex._emissions import make_action_event


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
    def test_chapter_rendered_as_assistant_message(self):
        events = [
            ChapterEvent(
                agent_name="t",
                name="Exploration",
                message="Found stuff",
            )
        ]
        messages = render_events_as_markdown(events)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
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
            make_action_event(
                agent_name="t", thinking="think", code="x = 1", title="Next step"
            ),
            SuccessEvent(agent_name="t", result=42),
        ]
        messages = render_events_as_markdown(events)
        # TaskStart(user), Chapter+Action(assistant), Success(user)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert "Go" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert "Early work" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "42" in messages[2]["content"]

    def test_multiple_chapters(self):
        events = [
            ChapterEvent(agent_name="t", name="Phase 1", message="Setup"),
            ChapterEvent(agent_name="t", name="Phase 2", message="Analysis"),
        ]
        messages = render_events_as_markdown(events)
        # Two consecutive assistant messages collapsed into one
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "Phase 1" in messages[0]["content"]
        assert "Phase 2" in messages[0]["content"]


class TestChapterEventXMLRendering:
    def test_chapter_rendered_as_assistant_message(self):
        events = [
            ChapterEvent(
                agent_name="t",
                name="Exploration",
                message="Found stuff",
            )
        ]
        messages = render_events_as_xml(events)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "Exploration" in messages[0]["content"]

    def test_chapter_in_xml_sequence(self):
        events = [
            ChapterEvent(agent_name="t", name="Setup", message="Done"),
            make_action_event(
                agent_name="t", thinking="think", code="x = 1", title="Act"
            ),
            FailEvent(agent_name="t", message="oops"),
        ]
        messages = render_events_as_xml(events)
        # Chapter+Action collapsed(assistant); FailEvent not rendered
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "Setup" in messages[0]["content"]
