"""Tests for ChaptersVFS (agex/fs/chapters_vfs.py)."""

from kvgit import store as kvgit_store

from agex.agent.events import (
    ActionEvent,
    ChapterEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.fs.chapters_vfs import build_chapters_dict, create_chapters_fs
from agex.state import _agex_decoder, _agex_encoder
from agex.state.log import add_event_to_log, replace_events_with_chapters


def _make_state():
    return kvgit_store(encoder=_agex_encoder, decoder=_agex_decoder)


def _chapter_with_events(state, name, message, events):
    """Helper: add events to state, create chapter via replace_events_with_chapters, return it."""
    for e in events:
        add_event_to_log(state, e)
    n = len(events)
    refs = state.get("__event_log__", [])
    start = len(refs) - n
    ch = ChapterEvent(agent_name="t", name=name, message=message)
    replace_events_with_chapters(state, [(start, start + n, ch)])
    # Return the chapter from the log
    log_events_after = [state.get(r) for r in state.get("__event_log__", [])]
    for e in log_events_after:
        if isinstance(e, ChapterEvent) and e.name == name:
            return e
    return ch


class TestBuildChaptersDict:
    def test_no_chapters(self):
        events = [
            ActionEvent(agent_name="t", thinking="t", code="x = 1"),
            SuccessEvent(agent_name="t", result=42),
        ]
        result = build_chapters_dict(events)
        assert result == {}

    def test_empty_events(self):
        assert build_chapters_dict([]) == {}

    def test_single_chapter(self):
        state = _make_state()
        e1 = ActionEvent(
            agent_name="t", thinking="explored", code="df.head()", title="Read"
        )
        ch = _chapter_with_events(state, "Data exploration", "Found 3 tables", [e1])
        result = build_chapters_dict([ch], state)

        assert "data-exploration/summary.md" in result
        assert "data-exploration/events/001-action.md" in result

        summary = result["data-exploration/summary.md"].decode()
        assert "# Data exploration" in summary
        assert "Found 3 tables" in summary

    def test_multiple_chapters(self):
        ch1 = ChapterEvent(agent_name="t", name="Phase One", message="Setup done")
        ch2 = ChapterEvent(agent_name="t", name="Phase Two", message="Analysis done")
        result = build_chapters_dict([ch1, ch2])

        assert "phase-one/summary.md" in result
        assert "phase-two/summary.md" in result

    def test_nested_chapters(self):
        state = _make_state()
        inner_event = ActionEvent(agent_name="t", thinking="t", code="x")
        add_event_to_log(state, inner_event)

        # Create inner chapter
        inner = ChapterEvent(agent_name="t", name="Inner Work", message="Details here")
        replace_events_with_chapters(state, [(0, 1, inner)])

        # Create outer chapter wrapping the inner
        outer = ChapterEvent(agent_name="t", name="Outer Work", message="Overview")
        replace_events_with_chapters(state, [(0, 1, outer)])

        log = [state.get(r) for r in state.get("__event_log__", [])]
        outer_ch = log[0]

        result = build_chapters_dict([outer_ch], state)

        # Outer chapter summary
        assert "outer-work/summary.md" in result
        # Inner chapter nested under outer
        assert "outer-work/chapters/inner-work/summary.md" in result
        assert "outer-work/chapters/inner-work/events/001-action.md" in result

    def test_non_chapter_events_ignored(self):
        events = [
            ActionEvent(agent_name="t", thinking="t", code="x"),
            ChapterEvent(agent_name="t", name="Work", message="Done"),
            SuccessEvent(agent_name="t", result=42),
        ]
        result = build_chapters_dict(events)
        # Only the chapter produces entries
        assert len(result) == 1  # just summary.md (no nested events — no refs)

    def test_slugification(self):
        ch = ChapterEvent(
            agent_name="t",
            name="Data Exploration & Analysis!",
            message="Done",
        )
        result = build_chapters_dict([ch])
        # Should slugify to something safe
        paths = list(result.keys())
        assert any("summary.md" in p for p in paths)
        # No special characters in paths
        for path in paths:
            assert "&" not in path
            assert "!" not in path

    def test_chapter_with_multiple_event_types(self):
        state = _make_state()
        events_inside = [
            TaskStartEvent(agent_name="t", task_name="sub", inputs={}, message="msg"),
            ActionEvent(agent_name="t", thinking="t", code="x", title="Step"),
            OutputEvent(agent_name="t", parts=[]),
            SuccessEvent(agent_name="t", result="ok"),
        ]
        ch = _chapter_with_events(
            state, "Mixed events", "Various event types", events_inside
        )
        result = build_chapters_dict([ch], state)
        assert "mixed-events/summary.md" in result
        assert "mixed-events/events/001-taskstart.md" in result
        assert "mixed-events/events/002-action.md" in result
        assert "mixed-events/events/003-output.md" in result
        assert "mixed-events/events/004-success.md" in result


class TestCreateChaptersFS:
    def test_returns_none_without_chapters(self):
        events = [ActionEvent(agent_name="t", thinking="t", code="x")]
        assert create_chapters_fs(events) is None

    def test_returns_none_for_empty(self):
        assert create_chapters_fs([]) is None

    def test_creates_readable_fs(self):
        state = _make_state()
        e1 = ActionEvent(agent_name="t", thinking="t", code="x")
        ch = _chapter_with_events(state, "Exploration", "Found stuff", [e1])
        fs = create_chapters_fs([ch], state)
        assert fs is not None

        # Can read summary
        content = fs.read("/exploration/summary.md")
        assert b"Exploration" in content
        assert b"Found stuff" in content

        # Can read events
        event_content = fs.read("/exploration/events/001-action.md")
        assert len(event_content) > 0

        # Directory structure exists
        assert fs.isdir("/exploration")
        assert fs.isdir("/exploration/events")
        assert fs.isfile("/exploration/summary.md")

    def test_read_only(self):
        ch = ChapterEvent(agent_name="t", name="Test", message="Msg")
        fs = create_chapters_fs([ch])
        assert fs is not None

        # Writing should fail (read-only)
        import pytest

        with pytest.raises(PermissionError):
            fs.write("/test/summary.md", b"hacked")

    def test_mountfs_integration(self):
        """Test that chapters FS can be mounted on a MountFS."""
        from monkeyfs import MountFS, VirtualFS

        base = VirtualFS()
        base.write("/app/main.py", b'print("hello")')

        mfs = MountFS(base)

        ch = ChapterEvent(
            agent_name="t",
            name="Setup",
            message="Initial setup done",
        )
        overlay = create_chapters_fs([ch])
        mfs.mount("/chapters", overlay)

        # Base files accessible
        assert mfs.exists("/app/main.py")
        # Chapter files accessible
        assert mfs.exists("/chapters/setup/summary.md")
        content = mfs.read("/chapters/setup/summary.md")
        assert b"Initial setup done" in content
