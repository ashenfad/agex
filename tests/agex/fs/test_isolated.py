"""Tests for IsolatedFS - secure filesystem with path restriction."""

import pytest
from kvit import Live, Staged, Versioned

from agex import Agent, connect_fs, pprint_events
from agex.fs import IsolatedFS
from agex.llm import Dummy, LLMResponse
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def _make_state():
    return Staged(Versioned(Memory()), encoder=_agex_encoder, decoder=_agex_decoder)


class TestIsolatedFSPathValidation:
    """Test path validation and security."""

    def test_creates_nonexistent_root(self, tmp_path):
        """Non-existent root directory should be automatically created."""
        new_root = tmp_path / "new_dir"
        assert not new_root.exists()

        IsolatedFS(root=str(new_root), state=Live())

        assert new_root.exists()
        assert new_root.is_dir()

    def test_rejects_relative_root(self):
        """Root must be absolute path."""
        with pytest.raises(ValueError, match="absolute path"):
            IsolatedFS(root="relative/path", state=Live())

    def test_rejects_file_as_root(self, tmp_path):
        """Root must be a directory, not a file."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("data")

        with pytest.raises(ValueError, match="must be a directory"):
            IsolatedFS(root=str(file_path), state=Live())

    def test_path_traversal_blocked(self, tmp_path):
        """Path traversal attempts should be blocked."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        # Try various path traversal attacks
        with pytest.raises(PermissionError, match="outside root"):
            fs.open("../../etc/passwd")

        with pytest.raises(PermissionError, match="outside root"):
            fs.read("../../../etc/passwd")

        with pytest.raises(PermissionError, match="outside root"):
            fs.write("../../../../tmp/evil.txt", b"data")

    def test_absolute_paths_are_rerooted(self, tmp_path):
        """Absolute paths are treated as relative to the isolated root."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        # Create a file at $ROOT/foo.txt
        (tmp_path / "foo.txt").write_text("data")

        # Open valid re-rooted absolute path
        # /foo.txt -> $ROOT/foo.txt
        with fs.open("/foo.txt", "r") as f:
            assert f.read() == "data"

        # Open invalid re-rooted path
        # /etc/passwd -> $ROOT/etc/passwd (which likely handles missing file)
        with pytest.raises(FileNotFoundError):
            fs.open("/etc/passwd")

        # But path traversal should STILL be blocked
        # /../../etc/passwd -> $ROOT/../etc/passwd -> OUTSIDE
        with pytest.raises(PermissionError, match="outside root"):
            fs.open("/../../etc/passwd")

    def test_symlink_to_outside_blocked(self, tmp_path):
        """Symlinks pointing outside root should be blocked."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        # Create symlink inside root pointing to outside
        link_path = tmp_path / "evil_link"
        link_path.symlink_to("/etc/passwd")

        with pytest.raises(PermissionError, match="outside root"):
            fs.read("evil_link")

    def test_symlink_to_inside_allowed(self, tmp_path):
        """Symlinks within root should work."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        # Create file and symlink within root
        (tmp_path / "data.txt").write_text("content")
        link_path = tmp_path / "link.txt"
        link_path.symlink_to(tmp_path / "data.txt")

        # Should be able to read through symlink
        content = fs.read("link.txt")
        assert content == b"content"


class TestIsolatedFSBasicOperations:
    """Test basic file operations."""

    def test_write_and_read(self, tmp_path):
        """Write and read files."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        fs.write("test.txt", b"Hello, world!")
        content = fs.read("test.txt")

        assert content == b"Hello, world!"

    def test_open_read_mode(self, tmp_path):
        """Open file in read mode."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        (tmp_path / "data.txt").write_text("test data")

        with fs.open("data.txt", "r") as f:
            content = f.read()

        assert content == "test data"

    def test_open_write_mode(self, tmp_path):
        """Open file in write mode."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        with fs.open("output.txt", "w") as f:
            f.write("new content")

        assert (tmp_path / "output.txt").read_text() == "new content"

    def test_exists(self, tmp_path):
        """Check if files exist."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        assert not fs.exists("missing.txt")

        (tmp_path / "present.txt").write_text("data")
        assert fs.exists("present.txt")

    def test_isfile_isdir(self, tmp_path):
        """Check file vs directory."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        (tmp_path / "file.txt").write_text("data")
        (tmp_path / "subdir").mkdir()

        assert fs.isfile("file.txt")
        assert not fs.isdir("file.txt")

        assert fs.isdir("subdir")
        assert not fs.isfile("subdir")

    def test_listdir(self, tmp_path):
        """List directory contents."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        (tmp_path / "file1.txt").write_text("data")
        (tmp_path / "file2.txt").write_text("data")
        (tmp_path / "subdir").mkdir()

        contents = fs.listdir(".")
        assert set(contents) == {"file1.txt", "file2.txt", "subdir"}

    def test_mkdir(self, tmp_path):
        """Create directories."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        fs.mkdir("newdir")
        assert (tmp_path / "newdir").is_dir()

        fs.mkdir("nested/deep/dir", parents=True)
        assert (tmp_path / "nested" / "deep" / "dir").is_dir()

    def test_remove(self, tmp_path):
        """Remove files."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        (tmp_path / "delete_me.txt").write_text("data")
        assert fs.exists("delete_me.txt")

        fs.remove("delete_me.txt")
        assert not fs.exists("delete_me.txt")

    def test_rename(self, tmp_path):
        """Rename files."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        (tmp_path / "old.txt").write_text("data")
        fs.rename("old.txt", "new.txt")

        assert not fs.exists("old.txt")
        assert fs.exists("new.txt")
        assert fs.read("new.txt") == b"data"

    def test_stat(self, tmp_path):
        """Get file metadata."""
        fs = IsolatedFS(root=str(tmp_path), state=Live())

        (tmp_path / "file.txt").write_bytes(b"12345")

        metadata = fs.stat("file.txt")
        assert metadata.size == 5
        assert metadata.created_at is not None
        assert metadata.modified_at is not None


class TestIsolatedFSTracking:
    """Test metadata tracking."""

    def test_tracking_disabled_no_state_updates(self, tmp_path):
        """When tracking=False, no state updates occur."""
        state = _make_state()
        fs = IsolatedFS(root=str(tmp_path), state=Live())  # Separate state, no tracking

        fs.write("file.txt", b"data")

        # State should not have metadata (using different state)
        assert state.get(fs.METADATA_KEY) is None

    def test_tracking_enabled_updates_metadata(self, tmp_path):
        """When tracking=True, metadata is updated."""
        state = _make_state()
        fs = IsolatedFS(root=str(tmp_path), state=state)

        fs.write("file.txt", b"data")

        # Metadata should be stored
        metadata = fs.get_metadata_snapshot()
        assert "file.txt" in metadata
        assert metadata["file.txt"].size == 4

    def test_tracking_detects_file_changes(self, tmp_path):
        """Tracking detects when files are modified."""
        state = _make_state()
        fs = IsolatedFS(root=str(tmp_path), state=state)

        fs.write("file.txt", b"original")
        snapshot1 = fs.get_metadata_snapshot()

        fs.write("file.txt", b"modified")
        snapshot2 = fs.get_metadata_snapshot()

        assert snapshot1["file.txt"].modified_at != snapshot2["file.txt"].modified_at


class TestIsolatedFSAgentIntegration:
    """Test isolated FS with agents."""

    def test_agent_with_isolated_fs(self, tmp_path):
        """Agent can use isolated filesystem."""
        # from agex import pprint_events
        # from agex.helpers import register_io

        # Create test workspace
        (tmp_path / "input.txt").write_text("Hello")

        llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="I'll read and transform the file",
                    code="""
f = open('input.txt', 'r')
data = f.read()
f.close()
with open('output.txt', 'w') as f2:
    f2.write(data.upper())
task_success('done')
""",
                )
            ]
        )

        agent = Agent(
            llm=llm,
            fs=connect_fs(type="isolated", root=str(tmp_path), tracking=False),
        )
        # swap_agent_fs_functions will be called automatically and will call register_io
        # register_io(agent)

        @agent.task
        def process() -> str:
            """Process files."""
            pass

        result = process(on_event=pprint_events)

        assert result == "done"
        assert (tmp_path / "output.txt").read_text() == "HELLO"

    def test_agent_cannot_escape_root(self, tmp_path):
        """Agent cannot access files outside root."""
        llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Try to escape",
                    code="open('/etc/passwd', 'r')",
                ),
                LLMResponse(
                    thinking="That failed, I'll give up",
                    code="task_success('failed to escape')",
                ),
            ]
        )

        agent = Agent(
            llm=llm,
            fs=connect_fs(type="isolated", root=str(tmp_path)),
        )

        @agent.task
        def escape_attempt() -> str:
            """Try to escape."""
            pass

        result = escape_attempt()

        # Agent should see permission error and handle it
        assert "failed to escape" in result.lower()

    def test_agent_with_tracking_emits_events(self, tmp_path):
        """Agent with tracking=True emits FileEvents."""
        from agex import connect_state, events
        from agex.agent.events import FileEvent

        llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Create files",
                    code="""
with open('new_file.txt', 'w') as f:
    f.write('created')
task_success('done')
""",
                )
            ]
        )

        state_config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            llm=llm,
            state=state_config,
            fs=connect_fs(type="isolated", root=str(tmp_path), tracking=True),
        )

        @agent.task
        def create_file() -> str:
            """Create a file."""
            pass

        _result = create_file()

        # Check that FileEvent was emitted
        state = agent.state()
        event_list = list(events(state))
        file_events = [e for e in event_list if isinstance(e, FileEvent)]

        assert len(file_events) > 0
        assert "new_file.txt" in file_events[0].added
