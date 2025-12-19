"""Tests for XML rendering utilities."""

from agex.agent.events import (
    ActionEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.render.xml import render_events_as_xml


class TestRenderEventsAsXML:
    """Tests for render_events_as_xml()."""

    def test_task_start_event(self):
        """Test rendering TaskStartEvent."""
        events = [
            TaskStartEvent(
                agent_name="test_agent",
                task_name="test_task",
                inputs={},
                message="Calculate sum of [1, 2, 3]",
            )
        ]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Calculate sum of [1, 2, 3]"

    def test_action_event(self):
        """Test rendering ActionEvent."""
        events = [
            ActionEvent(
                agent_name="test_agent",
                title="Summing numbers",
                thinking="I'll use sum() function",
                code="result = sum([1, 2, 3])\ntask_success(result)",
            )
        ]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        content = messages[0]["content"]
        assert "<TITLE>" in content
        assert "Summing numbers" in content
        assert "</TITLE>" in content
        assert "<THINKING>" in content
        assert "</THINKING>" in content
        assert "<PYTHON>" in content
        assert "</PYTHON>" in content
        assert "I'll use sum() function" in content
        assert "result = sum([1, 2, 3])" in content

    def test_output_event_text_only(self):
        """Test rendering OutputEvent with text only."""
        events = [OutputEvent(agent_name="test_agent", parts=["6"])]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "6" in messages[0]["content"]

    def test_success_event(self):
        """Test rendering SuccessEvent."""
        events = [SuccessEvent(agent_name="test_agent", result=6)]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "✅ Task completed:" in messages[0]["content"]
        assert "6" in messages[0]["content"]

    def test_fail_event(self):
        """Test rendering FailEvent."""
        events = [FailEvent(agent_name="test_agent", message="Invalid input")]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "❌ Task failed:" in messages[0]["content"]
        assert "Invalid input" in messages[0]["content"]

    def test_multiple_events(self):
        """Test rendering multiple events in sequence."""
        events = [
            TaskStartEvent(
                agent_name="test_agent",
                task_name="test_task",
                inputs={},
                message="Do a task",
            ),
            ActionEvent(
                agent_name="test_agent",
                title="Executing plan",
                thinking="My plan",
                code="do_something()",
            ),
            OutputEvent(agent_name="test_agent", parts=["Result"]),
            SuccessEvent(agent_name="test_agent", result="done"),
        ]
        messages = render_events_as_xml(events)

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"

    def test_system_note_event(self):
        """Test rendering SystemNoteEvent (regression test)."""
        from agex.agent.events import SystemNoteEvent

        events = [SystemNoteEvent(agent_name="System", message="FOREFRONT NOTE")]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "FOREFRONT NOTE"
