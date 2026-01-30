"""Tests for VFS size limiting."""

import pytest

from agex.fs import VirtualFS, connect_fs
from agex.state import Live


class TestVFSSizeLimit:
    """Tests for VirtualFS max_size_mb limit."""

    def test_no_limit_allows_any_size(self):
        """Test that no limit allows any file size."""
        state = Live()
        vfs = VirtualFS(state)  # No limit

        # Write 10MB file - should succeed
        content = b"x" * (10 * 1024 * 1024)
        vfs.write("/large.bin", content)

        assert vfs.read("/large.bin") == content

    def test_limit_allows_within_budget(self):
        """Test that writes within limit succeed."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.5MB - should succeed
        content = b"x" * (500 * 1024)
        vfs.write("/file.bin", content)

        assert vfs.read("/file.bin") == content

    def test_limit_blocks_oversized_single_file(self):
        """Test that a single file exceeding limit is rejected."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Try to write 2MB - should fail
        content = b"x" * (2 * 1024 * 1024)
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/large.bin", content)

    def test_limit_blocks_cumulative_overflow(self):
        """Test that cumulative writes exceeding limit are rejected."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.5MB - should succeed
        vfs.write("/file1.bin", b"x" * (500 * 1024))

        # Write another 0.6MB - should fail (total would be 1.1MB > 1MB)
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/file2.bin", b"y" * (600 * 1024))

    def test_overwrite_allows_same_size(self):
        """Test that overwriting with same size succeeds."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.5MB
        vfs.write("/file.bin", b"x" * (500 * 1024))

        # Overwrite with same size - should succeed
        vfs.write("/file.bin", b"y" * (500 * 1024))

        assert vfs.read("/file.bin") == b"y" * (500 * 1024)

    def test_overwrite_allows_smaller_size(self):
        """Test that overwriting with smaller size succeeds."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.5MB
        vfs.write("/file.bin", b"x" * (500 * 1024))

        # Overwrite with smaller - should succeed
        vfs.write("/file.bin", b"y" * (100 * 1024))

        assert len(vfs.read("/file.bin")) == 100 * 1024

    def test_remove_frees_space(self):
        """Test that removing files frees up space."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.6MB
        vfs.write("/file1.bin", b"x" * (600 * 1024))

        # Try to write 0.6MB more - should fail
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/file2.bin", b"y" * (600 * 1024))

        # Remove first file
        vfs.remove("/file1.bin")

        # Now second write should succeed
        vfs.write("/file2.bin", b"y" * (600 * 1024))
        assert vfs.read("/file2.bin") == b"y" * (600 * 1024)

    def test_write_many_respects_limit(self):
        """Test that write_many checks combined size."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Try to write multiple files that together exceed limit
        files = {
            "/a.bin": b"x" * (400 * 1024),
            "/b.bin": b"y" * (400 * 1024),
            "/c.bin": b"z" * (400 * 1024),  # Total 1.2MB > 1MB
        }
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write_many(files)

    def test_write_many_succeeds_within_limit(self):
        """Test that write_many succeeds when within limit."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write multiple files within limit
        files = {
            "/a.bin": b"x" * (300 * 1024),
            "/b.bin": b"y" * (300 * 1024),
        }
        vfs.write_many(files)

        assert vfs.read("/a.bin") == b"x" * (300 * 1024)
        assert vfs.read("/b.bin") == b"y" * (300 * 1024)


class TestConnectFsWithSizeLimit:
    """Tests for connect_fs with max_size_mb parameter."""

    def test_connect_fs_accepts_max_size_mb(self):
        """Test that connect_fs accepts max_size_mb parameter."""
        config = connect_fs(type="virtual", max_size_mb=50)
        assert config.max_size_mb == 50

    def test_connect_fs_default_is_unlimited(self):
        """Test that connect_fs defaults to unlimited."""
        config = connect_fs(type="virtual")
        assert config.max_size_mb is None
