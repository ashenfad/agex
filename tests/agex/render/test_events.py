"""Tests for markdown event rendering."""

from agex.agent.events import (
    ActionEvent,
    FileEvent,
    TaskStartEvent,
)
from agex.render.events import render_events_as_markdown


class TestRenderEventsAsMarkdown:
    """Tests for render_events_as_markdown()."""

    def test_file_event_markdown(self):
        """Test rendering FileEvent in markdown format."""
        events = [
            FileEvent(
                agent_name="test_agent",
                file_source="agent",
                added=["output.txt"],
                modified=["data.csv"],
                removed=["temp.tmp"],
            )
        ]
        messages = render_events_as_markdown(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert "[File changes by agent]" in content
        assert "output.txt" in content
        assert "data.csv" in content
        assert "temp.tmp" in content
        assert "Added:" in content
        assert "Modified:" in content
        assert "Removed:" in content

    def test_file_event_user_source(self):
        """Test FileEvent with user as source."""
        events = [
            FileEvent(
                agent_name="test_agent",
                file_source="user",
                added=["upload.csv"],
                modified=[],
                removed=[],
            )
        ]
        messages = render_events_as_markdown(events)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert "[File changes by user]" in content
        assert "upload.csv" in content
        assert "Added:" in content

    def test_file_event_in_sequence(self):
        """Test FileEvent rendering in sequence with other events."""
        events = [
            TaskStartEvent(
                agent_name="test_agent",
                task_name="test_task",
                inputs={},
                message="Process file",
            ),
            ActionEvent(
                agent_name="test_agent",
                thinking="I'll read the file",
                code="content = open('data.txt').read()",
            ),
            FileEvent(
                agent_name="test_agent",
                file_source="agent",
                added=[],
                modified=["data.txt"],
                removed=[],
            ),
        ]
        messages = render_events_as_markdown(events)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert "[File changes by agent]" in messages[2]["content"]
