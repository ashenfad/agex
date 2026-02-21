"""Tests for VirtualFS core functionality."""

import pytest
from gitkv import Live

from agex.fs import VirtualFS


class TestVirtualFSBasics:
    """Test basic VirtualFS operations."""

    def test_write_and_read_file(self):
        """Test writing and reading a file."""
        state = Live()
        vfs = VirtualFS(state)

        # Write
        vfs.write("test.txt", b"Hello, World!")

        # Read
        content = vfs.read("test.txt")
        assert content == b"Hello, World!"

    def test_file_not_found(self):
        """Test reading non-existent file raises FileNotFoundError."""
        state = Live()
        vfs = VirtualFS(state)

        with pytest.raises(FileNotFoundError):
            vfs.read("nonexistent.txt")

    def test_list_files(self):
        """Test listing files in root directory."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file1.txt", b"content1")
        vfs.write("file2.txt", b"content2")
        vfs.write("dir/file3.txt", b"content3")

        files = vfs.list("/")
        assert sorted(files) == ["dir", "file1.txt", "file2.txt"]

    def test_list_subdirectory(self):
        """Test listing files in subdirectory."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("data/file1.csv", b"a,b,c")
        vfs.write("data/file2.csv", b"x,y,z")
        vfs.write("other/file3.txt", b"text")

        files = vfs.list("data")
        assert sorted(files) == ["file1.csv", "file2.csv"]

    def test_exists(self):
        """Test checking file existence."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("exists.txt", b"content")

        assert vfs.exists("exists.txt") is True
        assert vfs.exists("nonexistent.txt") is False

    def test_exists_directory(self):
        """Test checking directory existence (implicit)."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("data/file.csv", b"content")

        assert vfs.exists("data") is True
        assert vfs.exists("nonexistent_dir") is False

    def test_isfile(self):
        """Test isfile checks."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"content")
        vfs.write("dir/nested.txt", b"nested")

        assert vfs.isfile("file.txt") is True
        assert vfs.isfile("dir") is False
        assert vfs.isfile("nonexistent") is False

    def test_isdir(self):
        """Test isdir checks (implicit directories)."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("dir/file.txt", b"content")

        assert vfs.isdir("dir") is True
        assert vfs.isdir("dir/file.txt") is False
        assert vfs.isdir("/") is True

    def test_getsize(self):
        """Test getting file size."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"Hello!")

        assert vfs.getsize("file.txt") == 6

    def test_remove(self):
        """Test removing a file."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("temp.txt", b"temp")
        assert vfs.exists("temp.txt") is True

        vfs.remove("temp.txt")
        assert vfs.exists("temp.txt") is False

    def test_remove_nonexistent(self):
        """Test removing non-existent file raises error."""
        state = Live()
        vfs = VirtualFS(state)

        with pytest.raises(FileNotFoundError):
            vfs.remove("nonexistent.txt")

    def test_rename(self):
        """Test renaming a file."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("old.txt", b"content")
        vfs.rename("old.txt", "new.txt")

        assert vfs.exists("old.txt") is False
        assert vfs.exists("new.txt") is True
        assert vfs.read("new.txt") == b"content"

    def test_write_append_mode(self):
        """Test writing in append mode."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"Line 1\n")
        vfs.write("file.txt", b"Line 2\n", mode="a")

        assert vfs.read("file.txt") == b"Line 1\nLine 2\n"

    def test_mkdir_creates_directory(self):
        """Test mkdir creates explicit directory entry."""
        state = Live()
        vfs = VirtualFS(state)

        # Should create directory
        vfs.mkdir("some_dir")

        # Should be accessible as a directory
        assert vfs.isdir("some_dir") is True
        assert vfs.exists("some_dir") is True
        assert vfs.isfile("some_dir") is False

    def test_makedirs_creates_tree(self):
        """Test makedirs creates parent directories."""
        state = Live()
        vfs = VirtualFS(state)

        # Creates entire tree
        vfs.makedirs("some/deep/path", exist_ok=True)

        # All levels should be directories
        assert vfs.isdir("some") is True
        assert vfs.isdir("some/deep") is True
        assert vfs.isdir("some/deep/path") is True


class TestVirtualFSPaths:
    """Test path handling in VirtualFS."""

    def test_path_normalization(self):
        """Test that leading slashes are handled correctly."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("/file.txt", b"content")
        assert vfs.read("file.txt") == b"content"
        assert vfs.exists("/file.txt") is True

    def test_nested_paths(self):
        """Test deeply nested paths."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("a/b/c/d/file.txt", b"deep")

        assert vfs.read("a/b/c/d/file.txt") == b"deep"
        assert vfs.exists("a/b/c/d") is True
        assert vfs.isdir("a/b/c") is True

    def test_path_encoding_roundtrip(self):
        """Test that path encoding/decoding is reversible."""
        state = Live()
        vfs = VirtualFS(state)

        paths = [
            "file.txt",
            "dir/file.txt",
            "deep/nested/path/file.csv",
            "special-chars_123.txt",
        ]

        for path in paths:
            encoded = vfs._encode_path(path)
            decoded = vfs._decode_path(encoded)
            assert decoded == path.lstrip("/") or decoded == "/"


class TestVirtualFSOpen:
    """Test VirtualFS.open() method."""

    def test_open_read_text(self):
        """Test opening file in text read mode."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"Hello, text!")

        with vfs.open("file.txt", "r") as f:
            content = f.read()

        assert content == "Hello, text!"
        assert isinstance(content, str)

    def test_open_read_binary(self):
        """Test opening file in binary read mode."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.bin", b"\x00\x01\x02\x03")

        with vfs.open("file.bin", "rb") as f:
            content = f.read()

        assert content == b"\x00\x01\x02\x03"
        assert isinstance(content, bytes)

    def test_open_write_text(self):
        """Test opening file in text write mode."""
        state = Live()
        vfs = VirtualFS(state)

        with vfs.open("file.txt", "w") as f:
            f.write("Hello, write!")

        assert vfs.read("file.txt") == b"Hello, write!"

    def test_open_write_binary(self):
        """Test opening file in binary write mode."""
        state = Live()
        vfs = VirtualFS(state)

        with vfs.open("file.bin", "wb") as f:
            f.write(b"\x04\x05\x06")

        assert vfs.read("file.bin") == b"\x04\x05\x06"

    def test_open_append_mode(self):
        """Test opening file in append mode."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"Line 1\n")

        with vfs.open("file.txt", "a") as f:
            f.write("Line 2\n")

        assert vfs.read("file.txt") == b"Line 1\nLine 2\n"

    def test_open_nonexistent_read(self):
        """Test opening non-existent file for reading raises error."""
        state = Live()
        vfs = VirtualFS(state)

        with pytest.raises(FileNotFoundError):
            vfs.open("nonexistent.txt", "r")


class TestVirtualFile:
    """Test VirtualFile class behavior."""

    def test_write_and_close(self):
        """Test that content is persisted on close."""
        state = Live()
        vfs = VirtualFS(state)

        f = vfs.open("file.txt", "w")
        f.write("content")
        # Not yet persisted
        assert vfs.exists("file.txt") is False

        f.close()
        # Now persisted
        assert vfs.read("file.txt") == b"content"

    def test_context_manager(self):
        """Test VirtualFile works as context manager."""
        state = Live()
        vfs = VirtualFS(state)

        with vfs.open("file.txt", "w") as f:
            f.write("auto-close")

        # File should be closed and persisted
        assert vfs.read("file.txt") == b"auto-close"

    def test_write_to_closed_file_raises(self):
        """Test writing to closed file raises error."""
        state = Live()
        vfs = VirtualFS(state)

        f = vfs.open("file.txt", "w")
        f.close()

        with pytest.raises(ValueError, match="closed file"):
            f.write("too late")

    def test_multiple_writes(self):
        """Test multiple writes to same file object."""
        state = Live()
        vfs = VirtualFS(state)

        with vfs.open("file.txt", "w") as f:
            f.write("Line 1\n")
            f.write("Line 2\n")
            f.write("Line 3\n")

        assert vfs.read("file.txt") == b"Line 1\nLine 2\nLine 3\n"
