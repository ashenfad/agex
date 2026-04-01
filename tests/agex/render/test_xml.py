"""Tests for XML rendering utilities."""

from agex.agent.datatypes import FileAction
from agex.agent.events import (
    ActionEvent,
    FailEvent,
    FileEvent,
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
        assert messages[0]["content"] == "[1] Calculate sum of [1, 2, 3]"

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

    def test_action_event_with_files(self):
        """Test rendering ActionEvent with files."""
        events = [
            ActionEvent(
                agent_name="test_agent",
                thinking="I'll create a file",
                code="import utils",
                file_actions=[
                    FileAction(path="utils.py", content="X = 1"),
                    FileAction(path="config.json", content="{}", mode="append"),
                ],
            )
        ]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert '<FILE path="utils.py">X = 1</FILE>' in content
        assert '<FILE path="config.json" mode="append">{}</FILE>' in content

    def test_output_event_text_only(self):
        """Test rendering OutputEvent with text only."""
        events = [OutputEvent(agent_name="test_agent", parts=["6"])]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "6" in messages[0]["content"]

    def test_success_event(self):
        """SuccessEvent renders the result repr in a TASK_SUCCESS tag."""
        events = [SuccessEvent(agent_name="test_agent", result=6)]
        messages = render_events_as_xml(events)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "<TASK_SUCCESS>" in messages[0]["content"]
        assert "6" in messages[0]["content"]

    def test_fail_event_not_rendered(self):
        """FailEvent is not rendered — intent is already in ActionEvent code."""
        events = [FailEvent(agent_name="test_agent", message="Invalid input")]
        messages = render_events_as_xml(events)
        assert len(messages) == 0

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

        # user(TaskStart), assistant(Action), user(Output), assistant(Success)
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert "<TASK_SUCCESS>" in messages[3]["content"]

    def test_system_note_event(self):
        """Test rendering SystemNoteEvent (regression test)."""
        from agex.agent.events import SystemNoteEvent

        events = [SystemNoteEvent(agent_name="System", message="FOREFRONT NOTE")]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "FOREFRONT NOTE"

    def test_file_event(self):
        """Test rendering FileEvent with XML tags."""
        events = [
            FileEvent(
                agent_name="test_agent",
                file_source="user",
                added=["file1.txt", "file2.txt"],
                modified=["file3.txt"],
                removed=[],
            )
        ]
        messages = render_events_as_xml(events)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert "<FILE_CHANGES" in content
        assert "</FILE_CHANGES>" in content
        assert "source='user'" in content
        assert "file1.txt" in content
        assert "file2.txt" in content
        assert "file3.txt" in content
        assert "Added:" in content
        assert "Modified:" in content
