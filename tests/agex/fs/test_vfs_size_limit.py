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

    def test_overwrite_allows_larger_within_limit(self):
        """Test that overwriting with larger size succeeds if still within limit."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.3MB
        vfs.write("/file.bin", b"x" * (300 * 1024))

        # Overwrite with larger but still under limit - should succeed
        vfs.write("/file.bin", b"y" * (800 * 1024))

        assert len(vfs.read("/file.bin")) == 800 * 1024

    def test_overwrite_blocks_exceeding_limit(self):
        """Test that overwriting that would exceed limit is blocked."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.3MB
        vfs.write("/file.bin", b"x" * (300 * 1024))

        # Try to overwrite with 1.5MB - should fail
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/file.bin", b"y" * (1500 * 1024))

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

    def test_remove_many_frees_space(self):
        """Test that remove_many frees up space."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write several small files
        vfs.write("/a.bin", b"x" * (200 * 1024))
        vfs.write("/b.bin", b"y" * (200 * 1024))
        vfs.write("/c.bin", b"z" * (200 * 1024))

        # Remove all
        vfs.remove_many(["/a.bin", "/b.bin", "/c.bin"])

        # Now should be able to write 0.9MB
        vfs.write("/new.bin", b"w" * (900 * 1024))
        assert len(vfs.read("/new.bin")) == 900 * 1024

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

    def test_write_many_atomic_on_limit_failure(self):
        """Test that write_many doesn't partially write on limit failure."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Try to write multiple files that together exceed limit
        files = {
            "/a.bin": b"x" * (400 * 1024),
            "/b.bin": b"y" * (400 * 1024),
            "/c.bin": b"z" * (400 * 1024),
        }
        with pytest.raises(OSError):
            vfs.write_many(files)

        # None of the files should exist
        assert not vfs.exists("/a.bin")
        assert not vfs.exists("/b.bin")
        assert not vfs.exists("/c.bin")

    def test_append_mode_respects_limit(self):
        """Test that append mode respects size limit."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.5MB
        vfs.write("/file.bin", b"x" * (500 * 1024))

        # Append 0.6MB - should fail (total would be 1.1MB)
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/file.bin", b"y" * (600 * 1024), mode="a")

    def test_append_mode_succeeds_within_limit(self):
        """Test that append mode succeeds when within limit."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.3MB
        vfs.write("/file.bin", b"x" * (300 * 1024))

        # Append 0.3MB - should succeed (total 0.6MB)
        vfs.write("/file.bin", b"y" * (300 * 1024), mode="a")

        content = vfs.read("/file.bin")
        assert len(content) == 600 * 1024


class TestVFSSizeLimitEdgeCases:
    """Edge case tests for VFS size limits."""

    def test_zero_limit_blocks_any_write(self):
        """Test that 0MB limit blocks any file write."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=0)

        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/file.bin", b"x")

    def test_very_small_limit(self):
        """Test behavior with very small limit."""
        state = Live()
        # 1KB limit
        vfs = VirtualFS(state, max_size_mb=0)  # Actually need fractional MB support

        # Since we can't do fractional MB, test with 1MB limit
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 999KB - should succeed
        vfs.write("/file.bin", b"x" * (999 * 1024))

        # Write another 100KB - should fail
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/file2.bin", b"y" * (100 * 1024))

    def test_exact_limit_boundary(self):
        """Test writing exactly at the limit boundary."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write exactly 1MB
        vfs.write("/file.bin", b"x" * (1024 * 1024))

        # Should not be able to write even 1 more byte in a new file
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/file2.bin", b"y")

    def test_multiple_small_files(self):
        """Test limit with many small files."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 10 files of 100KB each (total 1MB)
        for i in range(10):
            vfs.write(f"/file{i}.bin", b"x" * (100 * 1024))

        # 11th file should fail
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/file10.bin", b"y" * (100 * 1024))

    def test_size_limit_persists_across_operations(self):
        """Test that size tracking persists correctly across operations."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write, remove, write again
        vfs.write("/a.bin", b"x" * (400 * 1024))
        vfs.write("/b.bin", b"y" * (400 * 1024))
        vfs.remove("/a.bin")
        vfs.write("/c.bin", b"z" * (400 * 1024))

        # Now at 800KB, try to add 300KB more - should fail
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/d.bin", b"w" * (300 * 1024))

    def test_rename_doesnt_change_size(self):
        """Test that renaming a file doesn't affect size tracking."""
        state = Live()
        vfs = VirtualFS(state, max_size_mb=1)

        # Write 0.5MB
        vfs.write("/old.bin", b"x" * (500 * 1024))

        # Rename
        vfs.rename("/old.bin", "/new.bin")

        # Should still be able to write 0.4MB more
        vfs.write("/another.bin", b"y" * (400 * 1024))

        # But not 0.2MB more (would exceed 1MB)
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            vfs.write("/toomuch.bin", b"z" * (200 * 1024))


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

    def test_connect_fs_various_sizes(self):
        """Test connect_fs with various size values."""
        for size in [1, 10, 100, 1000]:
            config = connect_fs(type="virtual", max_size_mb=size)
            assert config.max_size_mb == size


class TestAgentVFSIntegration:
    """Tests for Agent integration with VFS size limits."""

    def test_agent_fs_respects_limit(self):
        """Test that Agent's VFS respects the size limit."""
        from agex import Agent, connect_fs, connect_llm, connect_state

        agent = Agent(
            llm=connect_llm(provider="dummy"),
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual", max_size_mb=1),
        )

        fs = agent.fs()

        # Write 0.5MB - should succeed
        fs.write("/file.bin", b"x" * (500 * 1024))

        # Write another 0.6MB - should fail
        with pytest.raises(OSError, match="VFS size limit exceeded"):
            fs.write("/file2.bin", b"y" * (600 * 1024))

    def test_agent_fs_unlimited_by_default(self):
        """Test that Agent's VFS is unlimited by default."""
        from agex import Agent, connect_llm, connect_state

        agent = Agent(
            llm=connect_llm(provider="dummy"),
            state=connect_state(type="live", storage="memory"),
        )

        fs = agent.fs()

        # Write 10MB - should succeed with default unlimited
        fs.write("/large.bin", b"x" * (10 * 1024 * 1024))
        assert len(fs.read("/large.bin")) == 10 * 1024 * 1024
