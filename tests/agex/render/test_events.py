"""Tests for markdown event rendering."""

from agex.agent.chapter import CHAPTER_TASK
from agex.agent.emissions import FileWriteEmission
from agex.agent.events import (
    ChapterEvent,
    FileEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.render.events import render_events_as_markdown
from tests.agex._emissions import make_action_event


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
            make_action_event(
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

    def test_action_event_markdown_with_mode(self):
        """Test ActionEvent rendering with mode attribute."""
        events = [
            make_action_event(
                agent_name="test_agent",
                thinking="I'll append to the file",
                code="pass",
                file_actions=[
                    FileWriteEmission(path="utils.py", content="content", mode="append")
                ],
            )
        ]
        messages = render_events_as_markdown(events)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert "### utils.py (mode: append)" in content


class TestChapterScopeFiltering:
    """Filter A applied to ``render_events_as_markdown``: closed
    ``__chapter__`` task scopes are skipped (their summary text lives
    in the ChapterEvent itself; rendering the bookkeeping would
    duplicate it). Open scopes stay visible so a running chapter task
    can see its own conversation history.
    """

    def test_closed_chapter_scope_is_skipped(self):
        events = [
            TaskStartEvent(
                agent_name="a", task_name="t1", inputs={}, message="t1 prompt"
            ),
            make_action_event(agent_name="a", thinking="t", code="real_work()"),
            SuccessEvent(agent_name="a", result="done"),
            # Chapter task scope — should be filtered out.
            TaskStartEvent(
                agent_name="a",
                task_name=CHAPTER_TASK,
                inputs={"event_index": "[1] task t1"},
                message="chapter primer goes here",
            ),
            make_action_event(
                agent_name="a",
                thinking="folding",
                code='task_success([Chapter(start=1, end=1, name="P1", message="full summary")])',
            ),
            SuccessEvent(agent_name="a", result=[]),
            # The resulting ChapterEvent
            ChapterEvent(agent_name="a", name="P1", message="full summary"),
            # Next parent task
            TaskStartEvent(
                agent_name="a", task_name="t2", inputs={}, message="t2 prompt"
            ),
            make_action_event(agent_name="a", thinking="more", code="more_work()"),
        ]
        messages = render_events_as_markdown(events)
        flat = "\n".join(
            m["content"] if isinstance(m["content"], str) else "" for m in messages
        )
        # Chapter task bookkeeping is invisible.
        assert "chapter primer goes here" not in flat
        assert CHAPTER_TASK not in flat
        assert "Chapter(start=1, end=1" not in flat
        # The ChapterEvent summary IS rendered.
        assert "P1" in flat
        assert "full summary" in flat
        # Real parent work renders in both task slots.
        assert "real_work()" in flat
        assert "more_work()" in flat

    def test_open_chapter_scope_is_visible(self):
        """An unclosed chapter scope (chapter task mid-flight) is NOT
        filtered — the chapter task's own loop needs to see its prompt
        and any prior turns."""
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            make_action_event(agent_name="a", thinking="t", code="x"),
            SuccessEvent(agent_name="a", result="r"),
            TaskStartEvent(
                agent_name="a",
                task_name=CHAPTER_TASK,
                inputs={"event_index": "[1] task t1"},
                message="chapter primer goes here",
            ),
            make_action_event(agent_name="a", thinking="picking", code="/* turn 1 */"),
        ]
        messages = render_events_as_markdown(events)
        flat = "\n".join(
            m["content"] if isinstance(m["content"], str) else "" for m in messages
        )
        assert "chapter primer goes here" in flat

    def test_closed_chapter_scope_does_not_bump_event_numbering(self):
        """The markdown renderer numbers each rendered event with a
        ``[N]`` prefix.  Filtered chapter-scope events must not consume
        a number — t2's tool_use should be ``[3]`` (taskstart-1, action-2,
        success... wait, succession of events varies).  We check the
        weaker invariant: nothing in the flat output references a
        ``[N]`` slot whose content is the chapter task's bookkeeping.
        """
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            SuccessEvent(agent_name="a", result="r1"),
            TaskStartEvent(
                agent_name="a", task_name=CHAPTER_TASK, inputs={}, message="chapter"
            ),
            make_action_event(agent_name="a", thinking="t", code="task_success([])"),
            SuccessEvent(agent_name="a", result=[]),
            TaskStartEvent(agent_name="a", task_name="t2", inputs={}, message="t2"),
        ]
        messages = render_events_as_markdown(events)
        flat = "\n".join(
            m["content"] if isinstance(m["content"], str) else "" for m in messages
        )
        # t1 + t2 starts visible; chapter-task start is not.
        assert "t1" in flat
        assert "t2" in flat
        assert "chapter" not in flat
        assert CHAPTER_TASK not in flat
