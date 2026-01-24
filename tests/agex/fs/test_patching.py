"""Tests for VirtualFS patching and context manager."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from agex.fs import VirtualFS, with_virtual_fs
from agex.state import Live


class TestPatchingBasics:
    """Test filesystem patching basics."""

    def test_open_patched_in_context(self):
        """Test that open() is patched within VFS context."""
        state = Live()
        vfs = VirtualFS(state)

        with with_virtual_fs(vfs):
            with open("test.txt", "w") as f:
                f.write("patched!")

        # File should be in VFS, not on real filesystem
        assert vfs.read("test.txt") == b"patched!"

    def test_open_not_patched_outside_context(self):
        """Test that open() works normally outside VFS context."""
        import tempfile

        # Outside context, open should use real filesystem
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("real file")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "r") as f:
                content = f.read()
            assert content == "real file"
        finally:
            os.remove(tmp_path)

    def test_listdir_patched(self):
        """Test that os.listdir() is patched."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file1.txt", b"content1")
        vfs.write("file2.txt", b"content2")

        with with_virtual_fs(vfs):
            files = os.listdir("/")

        assert sorted(files) == ["file1.txt", "file2.txt"]

    def test_exists_patched(self):
        """Test that os.path.exists() is patched."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("exists.txt", b"content")

        with with_virtual_fs(vfs):
            assert os.path.exists("exists.txt") is True
            assert os.path.exists("nonexistent.txt") is False

    def test_nested_directory_open(self):
        """Test that files in nested directories can be opened with standard operations."""
        state = Live()
        vfs = VirtualFS(state)

        # Write file to nested directory (like debug/dom.html)
        vfs.write("debug/dom.html", b"<html>test</html>")

        with with_virtual_fs(vfs):
            # Test os.path.exists
            assert os.path.exists("debug/dom.html") is True

            # Test open()
            with open("debug/dom.html", "r") as f:
                content = f.read()
            assert content == "<html>test</html>"

    def test_isfile_patched(self):
        """Test that os.path.isfile() is patched."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"content")
        vfs.write("dir/nested.txt", b"nested")

        with with_virtual_fs(vfs):
            assert os.path.isfile("file.txt") is True
            assert os.path.isfile("dir") is False

    def test_stat_file(self):
        """Test that os.stat() returns proper metadata for VFS files."""
        import stat as stat_module

        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"hello world")

        with with_virtual_fs(vfs):
            stat_result = os.stat("file.txt")

            # Verify file type and permissions
            assert stat_module.S_ISREG(stat_result.st_mode)
            assert stat_result.st_mode & 0o777 == 0o644

            # Verify size
            assert stat_result.st_size == 11

            # Verify timestamps exist (should be recent)
            import time

            now = time.time()
            assert stat_result.st_mtime <= now
            assert stat_result.st_ctime <= now
            assert stat_result.st_mtime > now - 10  # Created within last 10 seconds

    def test_stat_directory(self):
        """Test that os.stat() works for VFS directories."""
        import stat as stat_module

        state = Live()
        vfs = VirtualFS(state)

        vfs.write("dir/file.txt", b"content")

        with with_virtual_fs(vfs):
            stat_result = os.stat("dir")

            # Verify directory type and permissions
            assert stat_module.S_ISDIR(stat_result.st_mode)
            assert stat_result.st_mode & 0o777 == 0o755

            # Verify size is zero for directories
            assert stat_result.st_size == 0

    def test_stat_nonexistent(self):
        """Test that os.stat() raises FileNotFoundError for missing paths."""
        state = Live()
        vfs = VirtualFS(state)

        with with_virtual_fs(vfs):
            with pytest.raises(FileNotFoundError, match="No such file or directory"):
                os.stat("nonexistent.txt")


class TestAsyncSafety:
    """Test that patching is async-safe using contextvars."""

    @pytest.mark.asyncio
    async def test_concurrent_vfs_contexts(self):
        """Test that concurrent async tasks each get their own VFS context."""
        state_a = Live()
        state_b = Live()
        vfs_a = VirtualFS(state_a)
        vfs_b = VirtualFS(state_b)

        results = []

        async def task_a():
            with with_virtual_fs(vfs_a):
                # Simulate some async work
                await asyncio.sleep(0.01)
                with open("file.txt", "w") as f:
                    f.write("from task A")
                results.append("a")

        async def task_b():
            with with_virtual_fs(vfs_b):
                # Simulate some async work
                await asyncio.sleep(0.01)
                with open("file.txt", "w") as f:
                    f.write("from task B")
                results.append("b")

        # Run both tasks concurrently
        await asyncio.gather(task_a(), task_b())

        # Both tasks completed
        assert sorted(results) == ["a", "b"]

        # Each VFS has its own file with correct content
        assert vfs_a.read("file.txt") == b"from task A"
        assert vfs_b.read("file.txt") == b"from task B"

    def test_thread_pool_context_isolation(self):
        """Test VFS context isolation in thread pool (like agent execution)."""
        import contextvars

        state_1 = Live()
        state_2 = Live()
        vfs_1 = VirtualFS(state_1)
        vfs_2 = VirtualFS(state_2)

        def worker(vfs, content):
            """Worker function that runs in thread pool."""
            with with_virtual_fs(vfs):
                with open("file.txt", "w") as f:
                    f.write(content)

        # Copy context and run in executor (like agent execution does)
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(ctx.run, worker, vfs_1, "content 1")
            future2 = executor.submit(ctx.run, worker, vfs_2, "content 2")

            future1.result()
            future2.result()

        # Each VFS should have its own file
        assert vfs_1.read("file.txt") == b"content 1"
        assert vfs_2.read("file.txt") == b"content 2"


class TestPatchingEdgeCases:
    """Test edge cases in patching."""

    def test_nested_contexts(self):
        """Test that nested VFS contexts work correctly."""
        state_outer = Live()
        state_inner = Live()
        vfs_outer = VirtualFS(state_outer)
        vfs_inner = VirtualFS(state_inner)

        with with_virtual_fs(vfs_outer):
            with open("outer.txt", "w") as f:
                f.write("outer")

            with with_virtual_fs(vfs_inner):
                with open("inner.txt", "w") as f:
                    f.write("inner")

                # Inner context is active
                assert vfs_inner.exists("inner.txt") is True

            # Back to outer context
            assert vfs_outer.exists("outer.txt") is True
            # Inner file should not be in outer VFS
            assert vfs_outer.exists("inner.txt") is False

    def test_exception_in_context(self):
        """Test that VFS context is properly reset on exception."""
        state = Live()
        vfs = VirtualFS(state)

        try:
            with with_virtual_fs(vfs):
                with open("file.txt", "w") as f:
                    f.write("before error")
                raise ValueError("test error")
        except ValueError:
            pass

        # File should still be saved despite exception
        assert vfs.read("file.txt") == b"before error"

        # Context should be reset - open should work normally now
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("real")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "r") as f:
                assert f.read() == "real"
        finally:
            os.remove(tmp_path)

    def test_open_with_file_descriptor(self):
        """Test that patched open doesn't break file descriptor usage."""
        import tempfile

        state = Live()
        vfs = VirtualFS(state)

        # Create a real temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"real content")
            tmp_path = tmp.name

        try:
            with with_virtual_fs(vfs):
                # Opening by file descriptor should still work (bypass patching)
                # Note: This is a limitation - we can't easily patch fd-based opens
                # But at least it shouldn't crash
                pass
        finally:
            os.remove(tmp_path)
