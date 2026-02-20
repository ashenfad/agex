"""Tests for VirtualFS bulk operations (write_many, remove_many)."""

import pytest
from kvit import Live, Staged, Versioned

from agex.fs import VirtualFS
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def _make_state():
    return Staged(Versioned(Memory()), encoder=_agex_encoder, decoder=_agex_decoder)


class TestBulkOperations:
    """Test VirtualFS bulk operations."""

    def test_write_many_basic(self):
        """Test writing multiple files at once."""
        state = Live()
        vfs = VirtualFS(state)

        files = {
            "file1.txt": b"content 1",
            "file2.txt": b"content 2",
            "dir/file3.txt": b"content 3",
        }

        vfs.write_many(files)

        # All files should exist
        assert vfs.read("file1.txt") == b"content 1"
        assert vfs.read("file2.txt") == b"content 2"
        assert vfs.read("dir/file3.txt") == b"content 3"

    def test_write_many_validates_all_bytes(self):
        """Test that write_many validates all content is bytes."""
        state = Live()
        vfs = VirtualFS(state)

        files = {
            "file1.txt": b"content 1",
            "file2.txt": "not bytes",  # Invalid!
        }

        with pytest.raises(TypeError, match="Expected bytes for 'file2.txt'"):
            vfs.write_many(files)

        # No files should be written (validation happens first)
        assert not vfs.exists("file1.txt")

    def test_write_many_with_versioned_state_snapshots(self):
        """Test that write_many creates a single snapshot with Versioned state."""
        state = _make_state()
        vfs = VirtualFS(state)

        initial_history_length = len(list(state.versioned.history()))

        files = {
            "file1.txt": b"content 1",
            "file2.txt": b"content 2",
            "file3.txt": b"content 3",
        }

        vfs.write_many(files)

        # Should have created exactly one new commit (not 3)
        new_history_length = len(list(state.versioned.history()))
        assert new_history_length == initial_history_length + 1

    def test_write_many_with_live_state_no_snapshot(self):
        """Test that write_many works with Live state (no snapshot method)."""
        state = Live()
        vfs = VirtualFS(state)

        files = {
            "file1.txt": b"content 1",
            "file2.txt": b"content 2",
        }

        # Should not raise - Live state doesn't have snapshot()
        vfs.write_many(files)

        assert vfs.exists("file1.txt")
        assert vfs.exists("file2.txt")

    def test_remove_many_basic(self):
        """Test removing multiple files at once."""
        state = Live()
        vfs = VirtualFS(state)

        # Create files
        vfs.write("file1.txt", b"content 1")
        vfs.write("file2.txt", b"content 2")
        vfs.write("file3.txt", b"content 3")

        # Remove two of them
        vfs.remove_many(["file1.txt", "file2.txt"])

        # Removed files should not exist
        assert not vfs.exists("file1.txt")
        assert not vfs.exists("file2.txt")
        # Remaining file should still exist
        assert vfs.exists("file3.txt")

    def test_remove_many_validates_all_exist(self):
        """Test that remove_many validates all files exist first."""
        state = Live()
        vfs = VirtualFS(state)

        # Create one file
        vfs.write("file1.txt", b"content 1")

        # Try to remove two files (one doesn't exist)
        with pytest.raises(FileNotFoundError, match="file2.txt"):
            vfs.remove_many(["file1.txt", "file2.txt"])

        # First file should still exist (validation happens before removal)
        assert vfs.exists("file1.txt")

    def test_remove_many_multiple_missing_files(self):
        """Test error message with multiple missing files."""
        state = Live()
        vfs = VirtualFS(state)

        with pytest.raises(FileNotFoundError, match="Files not found"):
            vfs.remove_many(["file1.txt", "file2.txt", "file3.txt"])

    def test_remove_many_with_versioned_state_snapshots(self):
        """Test that remove_many creates a single snapshot with Versioned state."""
        state = _make_state()
        vfs = VirtualFS(state)

        # Create files
        vfs.write("file1.txt", b"content 1")
        state.commit()
        vfs.write("file2.txt", b"content 2")
        state.commit()
        vfs.write("file3.txt", b"content 3")
        state.commit()

        commits_before = len(list(state.versioned.history()))

        # Remove all three
        vfs.remove_many(["file1.txt", "file2.txt", "file3.txt"])

        # Should have created exactly one new commit (not 3)
        commits_after = len(list(state.versioned.history()))
        assert commits_after == commits_before + 1

    def test_remove_many_empty_list(self):
        """Test that removing empty list works."""
        state = Live()
        vfs = VirtualFS(state)

        # Should not raise
        vfs.remove_many([])
