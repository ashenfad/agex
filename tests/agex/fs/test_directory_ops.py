"""Tests for VirtualFS directory operations (mkdir, rmdir) and CWD (chdir, getcwd).

These tests verify the explicit directory support and current working directory
functionality added to VirtualFS.
"""

import os

import pytest
from kvit import Live, Staged, Versioned

from agex.fs import VirtualFS, with_virtual_fs
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def _make_state():
    return Staged(Versioned(Memory()), encoder=_agex_encoder, decoder=_agex_decoder)


class TestVFSMkdir:
    """Test VirtualFS.mkdir() functionality."""

    def test_mkdir_creates_directory(self):
        """mkdir creates an explicit directory entry."""
        vfs = VirtualFS(Live())

        vfs.mkdir("mydir")

        assert vfs.isdir("mydir") is True
        assert vfs.exists("mydir") is True
        assert vfs.isfile("mydir") is False

    def test_mkdir_with_leading_slash(self):
        """mkdir works with absolute paths."""
        vfs = VirtualFS(Live())

        vfs.mkdir("/mydir")

        assert vfs.isdir("mydir") is True
        assert vfs.isdir("/mydir") is True

    def test_mkdir_exist_ok_true(self):
        """mkdir with exist_ok=True doesn't raise on existing dir."""
        vfs = VirtualFS(Live())

        vfs.mkdir("mydir")
        vfs.mkdir("mydir", exist_ok=True)  # Should not raise

        assert vfs.isdir("mydir") is True

    def test_mkdir_exist_ok_false_raises(self):
        """mkdir with exist_ok=False raises on existing dir."""
        vfs = VirtualFS(Live())

        vfs.mkdir("mydir")

        with pytest.raises(FileExistsError):
            vfs.mkdir("mydir", exist_ok=False)

    def test_mkdir_raises_on_file(self):
        """mkdir raises if path is an existing file."""
        vfs = VirtualFS(Live())

        vfs.write("myfile", b"content")

        with pytest.raises(FileExistsError):
            vfs.mkdir("myfile")

    def test_mkdir_empty_dir_has_no_children(self):
        """Empty directory created by mkdir has no children."""
        vfs = VirtualFS(Live())

        vfs.mkdir("empty_dir")

        children = vfs.list("empty_dir")
        assert children == []


class TestVFSMakedirs:
    """Test VirtualFS.makedirs() functionality."""

    def test_makedirs_creates_tree(self):
        """makedirs creates entire directory tree."""
        vfs = VirtualFS(Live())

        vfs.makedirs("a/b/c")

        assert vfs.isdir("a") is True
        assert vfs.isdir("a/b") is True
        assert vfs.isdir("a/b/c") is True

    def test_makedirs_partial_tree_exists(self):
        """makedirs works when partial tree already exists."""
        vfs = VirtualFS(Live())

        vfs.mkdir("a")
        vfs.makedirs("a/b/c")

        assert vfs.isdir("a") is True
        assert vfs.isdir("a/b") is True
        assert vfs.isdir("a/b/c") is True

    def test_makedirs_raises_on_file_in_path(self):
        """makedirs raises if any path component is a file."""
        vfs = VirtualFS(Live())

        vfs.write("a", b"content")

        with pytest.raises(FileExistsError):
            vfs.makedirs("a/b/c")


class TestVFSRmdir:
    """Test VirtualFS.rmdir() functionality."""

    def test_rmdir_removes_empty_directory(self):
        """rmdir removes an empty directory."""
        vfs = VirtualFS(Live())

        vfs.mkdir("mydir")
        assert vfs.isdir("mydir") is True

        vfs.rmdir("mydir")

        assert vfs.exists("mydir") is False
        assert vfs.isdir("mydir") is False

    def test_rmdir_raises_on_non_empty_directory(self):
        """rmdir raises on non-empty directory."""
        vfs = VirtualFS(Live())

        vfs.write("mydir/file.txt", b"content")

        with pytest.raises(OSError, match="not empty"):
            vfs.rmdir("mydir")

    def test_rmdir_raises_on_file(self):
        """rmdir raises on file path."""
        vfs = VirtualFS(Live())

        vfs.write("myfile", b"content")

        with pytest.raises(NotADirectoryError):
            vfs.rmdir("myfile")

    def test_rmdir_raises_on_nonexistent(self):
        """rmdir raises on nonexistent path."""
        vfs = VirtualFS(Live())

        with pytest.raises(FileNotFoundError):
            vfs.rmdir("nonexistent")

    def test_rmdir_with_nested_empty_dirs(self):
        """rmdir only removes immediate directory, not parents."""
        vfs = VirtualFS(Live())

        vfs.makedirs("a/b/c")

        vfs.rmdir("a/b/c")

        assert vfs.isdir("a/b") is True
        assert vfs.isdir("a") is True
        assert vfs.exists("a/b/c") is False


class TestVFSChdir:
    """Test VirtualFS.chdir() and getcwd() functionality."""

    def test_getcwd_default_root(self):
        """getcwd returns / by default."""
        vfs = VirtualFS(Live())

        assert vfs.getcwd() == "/"

    def test_chdir_updates_cwd(self):
        """chdir updates the current working directory."""
        vfs = VirtualFS(Live())

        vfs.mkdir("mydir")
        vfs.chdir("mydir")

        assert vfs.getcwd() == "/mydir"

    def test_chdir_absolute_path(self):
        """chdir works with absolute paths."""
        vfs = VirtualFS(Live())

        vfs.makedirs("a/b/c")
        vfs.chdir("/a/b/c")

        assert vfs.getcwd() == "/a/b/c"

    def test_chdir_relative_path(self):
        """chdir works with relative paths."""
        vfs = VirtualFS(Live())

        vfs.makedirs("a/b/c")
        vfs.chdir("a")
        vfs.chdir("b")
        vfs.chdir("c")

        assert vfs.getcwd() == "/a/b/c"

    def test_chdir_parent_directory(self):
        """chdir works with .. to go up."""
        vfs = VirtualFS(Live())

        vfs.makedirs("a/b")
        vfs.chdir("a/b")
        vfs.chdir("..")

        assert vfs.getcwd() == "/a"

    def test_chdir_to_root(self):
        """chdir to / returns to root."""
        vfs = VirtualFS(Live())

        vfs.makedirs("a/b/c")
        vfs.chdir("a/b/c")
        vfs.chdir("/")

        assert vfs.getcwd() == "/"


class TestVFSCwdPathResolution:
    """Test that file operations respect current working directory."""

    def test_write_respects_cwd(self):
        """write resolves paths relative to CWD."""
        vfs = VirtualFS(Live())

        vfs.mkdir("mydir")
        vfs.chdir("mydir")
        vfs.write("file.txt", b"content")

        # File should be at /mydir/file.txt
        assert vfs.exists("/mydir/file.txt") is True
        assert vfs.read("/mydir/file.txt") == b"content"

    def test_read_respects_cwd(self):
        """read resolves paths relative to CWD."""
        vfs = VirtualFS(Live())

        vfs.write("mydir/file.txt", b"content")
        vfs.chdir("mydir")

        assert vfs.read("file.txt") == b"content"

    def test_exists_respects_cwd(self):
        """exists resolves paths relative to CWD."""
        vfs = VirtualFS(Live())

        vfs.write("mydir/file.txt", b"content")
        vfs.chdir("mydir")

        assert vfs.exists("file.txt") is True
        assert vfs.exists("nonexistent.txt") is False

    def test_list_respects_cwd(self):
        """list resolves paths relative to CWD."""
        vfs = VirtualFS(Live())

        vfs.write("mydir/file1.txt", b"1")
        vfs.write("mydir/file2.txt", b"2")
        vfs.chdir("mydir")

        files = vfs.list(".")
        assert sorted(files) == ["file1.txt", "file2.txt"]

    def test_remove_respects_cwd(self):
        """remove resolves paths relative to CWD."""
        vfs = VirtualFS(Live())

        vfs.write("mydir/file.txt", b"content")
        vfs.chdir("mydir")
        vfs.remove("file.txt")

        assert not vfs.exists("/mydir/file.txt")


class TestVFSOsPatches:
    """Test that os.* patched functions respect CWD."""

    def test_os_chdir_getcwd(self):
        """os.chdir and os.getcwd work with VFS."""
        vfs = VirtualFS(Live())
        vfs.mkdir("testdir")

        with with_virtual_fs(vfs):
            assert os.getcwd() == "/"

            os.chdir("testdir")
            assert os.getcwd() == "/testdir"

            os.chdir("/")
            assert os.getcwd() == "/"

    def test_os_mkdir_rmdir(self):
        """os.mkdir and os.rmdir work with VFS."""
        vfs = VirtualFS(Live())

        with with_virtual_fs(vfs):
            os.mkdir("newdir")
            assert os.path.isdir("newdir") is True

            os.rmdir("newdir")
            assert os.path.exists("newdir") is False

    def test_os_makedirs(self):
        """os.makedirs works with VFS."""
        vfs = VirtualFS(Live())

        with with_virtual_fs(vfs):
            os.makedirs("a/b/c", exist_ok=True)

            assert os.path.isdir("a") is True
            assert os.path.isdir("a/b") is True
            assert os.path.isdir("a/b/c") is True

    def test_os_path_abspath_respects_cwd(self):
        """os.path.abspath resolves against virtual CWD."""
        vfs = VirtualFS(Live())
        vfs.makedirs("mydir/subdir")

        with with_virtual_fs(vfs):
            assert os.path.abspath("file.txt") == "/file.txt"

            os.chdir("mydir")
            assert os.path.abspath("file.txt") == "/mydir/file.txt"
            assert os.path.abspath("../other.txt") == "/other.txt"


class TestVFSCwdPersistence:
    """Test that CWD persists across snapshots."""

    def test_cwd_persists_across_snapshots(self):
        """CWD is saved and restored with Versioned state."""
        state = _make_state()
        vfs = VirtualFS(state)

        vfs.makedirs("level1/level2")
        vfs.chdir("level1/level2")
        state.commit()

        # Create new VFS from same state
        vfs2 = VirtualFS(state)
        assert vfs2.getcwd() == "/level1/level2"

    def test_cwd_revert_on_checkout(self):
        """CWD is restored when checking out old state."""
        state = _make_state()
        vfs = VirtualFS(state)

        # State 1: CWD at root
        vfs.mkdir("dir1")
        state.commit()

        # State 2: CWD at /dir1
        vfs.chdir("dir1")
        state.commit()

        # Checkout state 1
        commits = list(state.versioned.history())
        old_state = state.checkout(commits[1])  # Second newest = first snapshot

        old_vfs = VirtualFS(old_state)
        assert old_vfs.getcwd() == "/"


class TestAutoCreateParentDirs:
    """Test auto-creation of parent directories on file write."""

    def test_write_creates_parent_dirs(self):
        """write auto-creates parent directories."""
        vfs = VirtualFS(Live())

        vfs.write("a/b/c/file.txt", b"content")

        assert vfs.isdir("a") is True
        assert vfs.isdir("a/b") is True
        assert vfs.isdir("a/b/c") is True
        assert vfs.isfile("a/b/c/file.txt") is True

    def test_write_with_cwd_creates_parents(self):
        """write in subdirectory auto-creates parents."""
        vfs = VirtualFS(Live())

        vfs.mkdir("workdir")
        vfs.chdir("workdir")
        vfs.write("sub/file.txt", b"content")

        assert vfs.isdir("/workdir/sub") is True
        assert vfs.isfile("/workdir/sub/file.txt") is True
