"""Tests for FileEvent emission."""

from agex.agent.events import FileEvent
from agex.fs import VirtualFS
from agex.state import Live
from agex.state.log import get_events_from_log


class TestFileEvent:
    """Test FileEvent data structure."""

    def test_file_event_creation(self):
        """Test basic FileEvent creation."""
        event = FileEvent(
            agent_name="test_agent",
            file_source="user",
            added=["file1.txt", "file2.txt"],
            modified=["file3.txt"],
            removed=[],
        )
        assert event.agent_name == "test_agent"
        assert event.file_source == "user"
        assert event.added == ["file1.txt", "file2.txt"]
        assert event.modified == ["file3.txt"]
        assert event.removed == []

    def test_file_event_str(self):
        """Test FileEvent string representation."""
        event = FileEvent(
            agent_name="test",
            file_source="agent",
            added=["a.txt"],
            modified=["b.txt"],
            removed=["c.txt"],
        )
        s = str(event)
        assert "agent" in s
        assert "1 added" in s
        assert "1 modified" in s
        assert "1 removed" in s

    def test_file_event_markdown(self):
        """Test FileEvent markdown representation."""
        event = FileEvent(
            agent_name="test",
            file_source="user",
            added=["new.txt"],
            modified=[],
            removed=[],
        )
        md = event._repr_markdown_()
        assert "Added" in md
        assert "new.txt" in md

    def test_file_event_html(self):
        """Test FileEvent HTML representation."""
        event = FileEvent(
            agent_name="test",
            file_source="agent",
            added=[],
            modified=["updated.txt"],
            removed=[],
        )
        html = event._repr_html_()
        assert "updated.txt" in html
        assert "Modified" in html


class TestAgentAwareVFSEvents:
    """Test that AgentAwareVFS emits correct FileEvents."""

    def test_write_emits_event_for_new_file(self):
        """Test that writing a new file emits FileEvent with added."""
        from agex.fs import AgentAwareVFS

        state = Live()
        vfs = VirtualFS(state)
        aware_vfs = AgentAwareVFS(vfs, state, "test_agent")

        aware_vfs.write("new.txt", b"content")

        events = get_events_from_log(state)
        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) == 1
        assert file_events[0].added == ["new.txt"]
        assert file_events[0].modified == []
        assert file_events[0].file_source == "user"

    def test_write_emits_event_for_modified_file(self):
        """Test that overwriting a file emits FileEvent with modified."""
        from agex.fs import AgentAwareVFS

        state = Live()
        vfs = VirtualFS(state)
        vfs.write("existing.txt", b"original")  # Create without AgentAware

        aware_vfs = AgentAwareVFS(vfs, state, "test_agent")
        aware_vfs.write("existing.txt", b"updated")

        events = get_events_from_log(state)
        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) == 1
        assert file_events[0].added == []
        assert file_events[0].modified == ["existing.txt"]

    def test_remove_emits_event(self):
        """Test that removing a file emits FileEvent with removed."""
        from agex.fs import AgentAwareVFS

        state = Live()
        vfs = VirtualFS(state)
        vfs.write("to_delete.txt", b"content")

        aware_vfs = AgentAwareVFS(vfs, state, "test_agent")
        aware_vfs.remove("to_delete.txt")

        events = get_events_from_log(state)
        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) == 1
        assert file_events[0].removed == ["to_delete.txt"]

    def test_write_many_emits_single_event(self):
        """Test that write_many emits one event for all files."""
        from agex.fs import AgentAwareVFS

        state = Live()
        vfs = VirtualFS(state)
        vfs.write("existing.txt", b"original")

        aware_vfs = AgentAwareVFS(vfs, state, "test_agent")
        aware_vfs.write_many(
            {
                "new1.txt": b"content1",
                "new2.txt": b"content2",
                "existing.txt": b"updated",
            }
        )

        events = get_events_from_log(state)
        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) == 1
        assert set(file_events[0].added) == {"new1.txt", "new2.txt"}
        assert file_events[0].modified == ["existing.txt"]

    def test_rename_emits_removed_and_added(self):
        """Test that rename emits both removed and added."""
        from agex.fs import AgentAwareVFS

        state = Live()
        vfs = VirtualFS(state)
        vfs.write("old.txt", b"content")

        aware_vfs = AgentAwareVFS(vfs, state, "test_agent")
        aware_vfs.rename("old.txt", "new.txt")

        events = get_events_from_log(state)
        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) == 1
        assert file_events[0].removed == ["old.txt"]
        assert file_events[0].added == ["new.txt"]

    def test_read_does_not_emit_event(self):
        """Test that read operations don't emit events."""
        from agex.fs import AgentAwareVFS

        state = Live()
        vfs = VirtualFS(state)
        vfs.write("file.txt", b"content")

        aware_vfs = AgentAwareVFS(vfs, state, "test_agent")
        aware_vfs.read("file.txt")
        aware_vfs.exists("file.txt")
        aware_vfs.list("/")

        events = get_events_from_log(state)
        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) == 0
