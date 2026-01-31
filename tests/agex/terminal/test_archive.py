"""
Tests for archive commands: tar, gzip, gunzip, zip, unzip.
"""

import gzip as gzip_module
import io
import tarfile
import zipfile

import pytest

from agex.fs.virtual import VirtualFS
from agex.state import Live
from agex.terminal import to_script
from agex.terminal.interpreter.core import execute_script
from agex.terminal.interpreter.datatypes import TerminalError


@pytest.fixture
def fs():
    """Create a VirtualFS for testing."""
    state = Live()
    return VirtualFS(state)


class TestGzip:
    """Tests for gzip command."""

    def test_gzip_compress_file(self, fs):
        """Test basic file compression."""
        fs.write("/workspace/test.txt", b"Hello, World!")

        script = to_script("gzip /workspace/test.txt")
        execute_script(script, fs)

        # Original should be removed
        assert not fs.exists("/workspace/test.txt")
        # Compressed file should exist
        assert fs.exists("/workspace/test.txt.gz")

        # Verify it's valid gzip
        compressed = fs.read("/workspace/test.txt.gz")
        decompressed = gzip_module.decompress(compressed)
        assert decompressed == b"Hello, World!"

    def test_gzip_keep_original(self, fs):
        """Test -k flag to keep original."""
        fs.write("/workspace/test.txt", b"Hello, World!")

        script = to_script("gzip -k /workspace/test.txt")
        execute_script(script, fs)

        # Both should exist
        assert fs.exists("/workspace/test.txt")
        assert fs.exists("/workspace/test.txt.gz")

    def test_gzip_decompress(self, fs):
        """Test gzip -d for decompression."""
        original = b"Hello, World!"
        compressed = gzip_module.compress(original)
        fs.write("/workspace/test.txt.gz", compressed)

        script = to_script("gzip -d /workspace/test.txt.gz")
        execute_script(script, fs)

        # Compressed should be removed
        assert not fs.exists("/workspace/test.txt.gz")
        # Decompressed file should exist
        assert fs.exists("/workspace/test.txt")
        assert fs.read("/workspace/test.txt") == original

    def test_gzip_no_overwrite_without_force(self, fs):
        """Test that gzip won't overwrite without -f."""
        fs.write("/workspace/test.txt", b"original")
        fs.write("/workspace/test.txt.gz", b"existing")

        script = to_script("gzip /workspace/test.txt")
        with pytest.raises(TerminalError, match="already exists"):
            execute_script(script, fs)

    def test_gzip_force_overwrite(self, fs):
        """Test -f flag for forced overwrite."""
        fs.write("/workspace/test.txt", b"new content")
        fs.write("/workspace/test.txt.gz", b"old compressed")

        script = to_script("gzip -f /workspace/test.txt")
        execute_script(script, fs)

        # Should have overwritten
        compressed = fs.read("/workspace/test.txt.gz")
        assert gzip_module.decompress(compressed) == b"new content"

    def test_gzip_already_compressed(self, fs):
        """Test error when trying to compress .gz file."""
        fs.write("/workspace/test.txt.gz", b"data")

        script = to_script("gzip /workspace/test.txt.gz")
        with pytest.raises(TerminalError, match="already has .gz suffix"):
            execute_script(script, fs)

    def test_gzip_file_not_found(self, fs):
        """Test error for missing file."""
        script = to_script("gzip /workspace/nonexistent.txt")
        with pytest.raises(TerminalError, match="No such file"):
            execute_script(script, fs)


class TestGunzip:
    """Tests for gunzip command."""

    def test_gunzip_basic(self, fs):
        """Test gunzip is equivalent to gzip -d."""
        original = b"Test content"
        compressed = gzip_module.compress(original)
        fs.write("/workspace/file.txt.gz", compressed)

        script = to_script("gunzip /workspace/file.txt.gz")
        execute_script(script, fs)

        assert not fs.exists("/workspace/file.txt.gz")
        assert fs.exists("/workspace/file.txt")
        assert fs.read("/workspace/file.txt") == original

    def test_gunzip_keep(self, fs):
        """Test gunzip -k to keep compressed file."""
        original = b"Test content"
        compressed = gzip_module.compress(original)
        fs.write("/workspace/file.txt.gz", compressed)

        script = to_script("gunzip -k /workspace/file.txt.gz")
        execute_script(script, fs)

        # Both should exist
        assert fs.exists("/workspace/file.txt.gz")
        assert fs.exists("/workspace/file.txt")


class TestTar:
    """Tests for tar command."""

    def test_tar_create_single_file(self, fs):
        """Test creating tar archive with single file."""
        fs.write("/workspace/file.txt", b"file content")

        script = to_script("tar -cf /workspace/archive.tar /workspace/file.txt")
        execute_script(script, fs)

        assert fs.exists("/workspace/archive.tar")

        # Verify tar contents
        content = fs.read("/workspace/archive.tar")
        with tarfile.open(fileobj=io.BytesIO(content), mode="r") as tf:
            names = tf.getnames()
            assert "/workspace/file.txt" in names

    def test_tar_create_multiple_files(self, fs):
        """Test creating tar archive with multiple files."""
        fs.write("/workspace/a.txt", b"file a")
        fs.write("/workspace/b.txt", b"file b")

        script = to_script(
            "tar -cf /workspace/archive.tar /workspace/a.txt /workspace/b.txt"
        )
        execute_script(script, fs)

        content = fs.read("/workspace/archive.tar")
        with tarfile.open(fileobj=io.BytesIO(content), mode="r") as tf:
            names = tf.getnames()
            assert "/workspace/a.txt" in names
            assert "/workspace/b.txt" in names

    def test_tar_create_directory(self, fs):
        """Test creating tar archive with directory."""
        fs.makedirs("/workspace/mydir")
        fs.write("/workspace/mydir/file1.txt", b"file 1")
        fs.write("/workspace/mydir/file2.txt", b"file 2")

        script = to_script("tar -cf /workspace/archive.tar /workspace/mydir")
        execute_script(script, fs)

        content = fs.read("/workspace/archive.tar")
        with tarfile.open(fileobj=io.BytesIO(content), mode="r") as tf:
            names = tf.getnames()
            assert any("mydir" in n for n in names)
            assert any("file1.txt" in n for n in names)
            assert any("file2.txt" in n for n in names)

    def test_tar_deeply_nested_directory(self, fs):
        """Test creating tar archive with deeply nested directory structure."""
        # Create 3-level nested structure
        fs.makedirs("/workspace/a/b/c")
        fs.write("/workspace/a/root.txt", b"root level")
        fs.write("/workspace/a/b/middle.txt", b"middle level")
        fs.write("/workspace/a/b/c/deep.txt", b"deep level")
        # Also add a sibling at level 2
        fs.makedirs("/workspace/a/d")
        fs.write("/workspace/a/d/sibling.txt", b"sibling")

        script = to_script("tar -cf /workspace/nested.tar /workspace/a")
        execute_script(script, fs)

        content = fs.read("/workspace/nested.tar")
        with tarfile.open(fileobj=io.BytesIO(content), mode="r") as tf:
            names = tf.getnames()
            # Verify all files are present
            assert any("root.txt" in n for n in names)
            assert any("middle.txt" in n for n in names)
            assert any("deep.txt" in n for n in names)
            assert any("sibling.txt" in n for n in names)
            # Verify the actual content
            deep_member = [m for m in tf.getmembers() if "deep.txt" in m.name][0]
            extracted = tf.extractfile(deep_member)
            assert extracted.read() == b"deep level"

    def test_tar_create_gzipped(self, fs):
        """Test creating gzipped tar archive."""
        fs.write("/workspace/file.txt", b"file content")

        script = to_script("tar -czf /workspace/archive.tar.gz /workspace/file.txt")
        execute_script(script, fs)

        # Should be valid gzipped tar
        content = fs.read("/workspace/archive.tar.gz")
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tf:
            names = tf.getnames()
            assert "/workspace/file.txt" in names

    def test_tar_extract(self, fs):
        """Test extracting tar archive."""
        # Create archive manually
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tf:
            info = tarfile.TarInfo(name="extracted/test.txt")
            content = b"extracted content"
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
        fs.write("/workspace/archive.tar", buffer.getvalue())

        script = to_script("tar -xf /workspace/archive.tar -C /workspace")
        execute_script(script, fs)

        assert fs.exists("/workspace/extracted/test.txt")
        assert fs.read("/workspace/extracted/test.txt") == b"extracted content"

    def test_tar_extract_gzipped(self, fs):
        """Test extracting gzipped tar archive."""
        # Create gzipped archive manually
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
            info = tarfile.TarInfo(name="test.txt")
            content = b"gzipped content"
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
        fs.write("/workspace/archive.tar.gz", buffer.getvalue())

        script = to_script("tar -xzf /workspace/archive.tar.gz -C /workspace")
        execute_script(script, fs)

        assert fs.exists("/workspace/test.txt")
        assert fs.read("/workspace/test.txt") == b"gzipped content"

    def test_tar_list(self, fs):
        """Test listing tar archive contents."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tf:
            for name in ["file1.txt", "file2.txt", "dir/file3.txt"]:
                info = tarfile.TarInfo(name=name)
                info.size = 0
                tf.addfile(info, io.BytesIO(b""))
        fs.write("/workspace/archive.tar", buffer.getvalue())

        script = to_script("tar -tf /workspace/archive.tar")
        output = execute_script(script, fs)

        assert "file1.txt" in output
        assert "file2.txt" in output
        assert "dir/file3.txt" in output

    def test_tar_verbose(self, fs):
        """Test verbose output during creation."""
        fs.write("/workspace/file.txt", b"content")

        script = to_script("tar -cvf /workspace/archive.tar /workspace/file.txt")
        output = execute_script(script, fs)

        assert "/workspace/file.txt" in output

    def test_tar_requires_mode(self, fs):
        """Test that tar requires exactly one mode flag."""
        script = to_script("tar -f /workspace/archive.tar")
        with pytest.raises(TerminalError, match="exactly one of -c, -x, -t"):
            execute_script(script, fs)

    def test_tar_requires_file(self, fs):
        """Test that tar requires -f flag."""
        script = to_script("tar -c /workspace/file.txt")
        with pytest.raises(TerminalError, match="-f option is required"):
            execute_script(script, fs)

    def test_tar_skips_macos_appledouble_files(self, fs):
        """Test that macOS AppleDouble resource fork files (._*) are skipped."""
        # Create archive with AppleDouble files (like macOS creates)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tf:
            # Regular file
            info = tarfile.TarInfo(name="app/main.py")
            content = b"print('hello')"
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))

            # AppleDouble resource fork (should be skipped)
            info = tarfile.TarInfo(name="app/._main.py")
            content = b"resource fork data"
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))

            # Another regular file
            info = tarfile.TarInfo(name="app/utils.py")
            content = b"def helper(): pass"
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))

            # Its AppleDouble (should be skipped)
            info = tarfile.TarInfo(name="app/._utils.py")
            content = b"resource fork data"
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))

        fs.write("/workspace/app.tar", buffer.getvalue())

        script = to_script("tar -xf /workspace/app.tar -C /workspace")
        execute_script(script, fs)

        # Regular files should exist
        assert fs.exists("/workspace/app/main.py")
        assert fs.exists("/workspace/app/utils.py")

        # AppleDouble files should NOT exist
        assert not fs.exists("/workspace/app/._main.py")
        assert not fs.exists("/workspace/app/._utils.py")

    def test_tar_path_traversal_blocked(self, fs):
        """Test that path traversal is blocked during extraction."""
        # Create archive with path traversal attempt
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tf:
            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = 0
            tf.addfile(info, io.BytesIO(b""))
        fs.write("/workspace/evil.tar", buffer.getvalue())

        script = to_script("tar -xf /workspace/evil.tar -C /workspace")
        with pytest.raises(TerminalError, match="path traversal"):
            execute_script(script, fs)


class TestZip:
    """Tests for zip command."""

    def test_zip_single_file(self, fs):
        """Test creating zip with single file."""
        fs.write("/workspace/file.txt", b"file content")

        script = to_script("zip /workspace/archive.zip /workspace/file.txt")
        execute_script(script, fs)

        assert fs.exists("/workspace/archive.zip")

        # Verify zip contents
        content = fs.read("/workspace/archive.zip")
        with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
            names = zf.namelist()
            assert "/workspace/file.txt" in names
            assert zf.read("/workspace/file.txt") == b"file content"

    def test_zip_multiple_files(self, fs):
        """Test creating zip with multiple files."""
        fs.write("/workspace/a.txt", b"file a")
        fs.write("/workspace/b.txt", b"file b")

        script = to_script("zip /workspace/archive /workspace/a.txt /workspace/b.txt")
        execute_script(script, fs)

        # Should add .zip if not present
        assert fs.exists("/workspace/archive.zip")

        content = fs.read("/workspace/archive.zip")
        with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
            names = zf.namelist()
            assert "/workspace/a.txt" in names
            assert "/workspace/b.txt" in names

    def test_zip_directory_requires_r(self, fs):
        """Test that zipping directory requires -r flag."""
        fs.makedirs("/workspace/mydir")
        fs.write("/workspace/mydir/file.txt", b"content")

        script = to_script("zip /workspace/archive.zip /workspace/mydir")
        with pytest.raises(TerminalError, match="is a directory"):
            execute_script(script, fs)

    def test_zip_directory_recursive(self, fs):
        """Test zipping directory with -r flag."""
        fs.makedirs("/workspace/mydir")
        fs.write("/workspace/mydir/file1.txt", b"file 1")
        fs.write("/workspace/mydir/file2.txt", b"file 2")

        script = to_script("zip -r /workspace/archive.zip /workspace/mydir")
        execute_script(script, fs)

        content = fs.read("/workspace/archive.zip")
        with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
            names = zf.namelist()
            assert any("mydir" in n for n in names)
            assert any("file1.txt" in n for n in names)
            assert any("file2.txt" in n for n in names)

    def test_zip_deeply_nested_recursive(self, fs):
        """Test zipping deeply nested directory structure with -r flag."""
        # Create 3-level nested structure
        fs.makedirs("/workspace/a/b/c")
        fs.write("/workspace/a/root.txt", b"root level")
        fs.write("/workspace/a/b/middle.txt", b"middle level")
        fs.write("/workspace/a/b/c/deep.txt", b"deep level")
        # Also add a sibling at level 2
        fs.makedirs("/workspace/a/d")
        fs.write("/workspace/a/d/sibling.txt", b"sibling")

        script = to_script("zip -r /workspace/nested.zip /workspace/a")
        execute_script(script, fs)

        content = fs.read("/workspace/nested.zip")
        with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
            names = zf.namelist()
            # Verify all files are present
            assert any("root.txt" in n for n in names)
            assert any("middle.txt" in n for n in names)
            assert any("deep.txt" in n for n in names)
            assert any("sibling.txt" in n for n in names)
            # Verify the actual content
            deep_file = [n for n in names if "deep.txt" in n][0]
            assert zf.read(deep_file) == b"deep level"


class TestUnzip:
    """Tests for unzip command."""

    def test_unzip_basic(self, fs):
        """Test basic zip extraction."""
        # Create zip manually
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("test.txt", b"test content")
        fs.write("/workspace/archive.zip", buffer.getvalue())

        script = to_script("unzip /workspace/archive.zip -d /workspace/output")
        output = execute_script(script, fs)

        assert fs.exists("/workspace/output/test.txt")
        assert fs.read("/workspace/output/test.txt") == b"test content"
        assert "inflating" in output

    def test_unzip_nested_directories(self, fs):
        """Test extracting zip with nested directories."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("dir1/dir2/file.txt", b"nested content")
        fs.write("/workspace/archive.zip", buffer.getvalue())

        script = to_script("unzip /workspace/archive.zip -d /workspace")
        execute_script(script, fs)

        assert fs.exists("/workspace/dir1/dir2/file.txt")
        assert fs.read("/workspace/dir1/dir2/file.txt") == b"nested content"

    def test_unzip_skips_macos_appledouble_files(self, fs):
        """Test that macOS AppleDouble resource fork files (._*) are skipped."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            # Regular files
            zf.writestr("app/main.py", b"print('hello')")
            zf.writestr("app/utils.py", b"def helper(): pass")
            # AppleDouble resource forks (should be skipped)
            zf.writestr("app/._main.py", b"resource fork data")
            zf.writestr("app/._utils.py", b"resource fork data")
        fs.write("/workspace/app.zip", buffer.getvalue())

        script = to_script("unzip /workspace/app.zip -d /workspace")
        execute_script(script, fs)

        # Regular files should exist
        assert fs.exists("/workspace/app/main.py")
        assert fs.exists("/workspace/app/utils.py")

        # AppleDouble files should NOT exist
        assert not fs.exists("/workspace/app/._main.py")
        assert not fs.exists("/workspace/app/._utils.py")

    def test_unzip_list(self, fs):
        """Test listing zip contents."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("file1.txt", b"content1")
            zf.writestr("file2.txt", b"content2content2")
        fs.write("/workspace/archive.zip", buffer.getvalue())

        script = to_script("unzip -l /workspace/archive.zip")
        output = execute_script(script, fs)

        assert "file1.txt" in output
        assert "file2.txt" in output
        assert "2 files" in output

    def test_unzip_skip_existing(self, fs):
        """Test that existing files are skipped without -o."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("existing.txt", b"new content")
        fs.write("/workspace/archive.zip", buffer.getvalue())
        fs.write("/workspace/existing.txt", b"old content")

        script = to_script("unzip /workspace/archive.zip -d /workspace")
        output = execute_script(script, fs)

        # Should skip and keep old content
        assert "skipping" in output
        assert fs.read("/workspace/existing.txt") == b"old content"

    def test_unzip_overwrite(self, fs):
        """Test -o flag for overwriting files."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("existing.txt", b"new content")
        fs.write("/workspace/archive.zip", buffer.getvalue())
        fs.write("/workspace/existing.txt", b"old content")

        script = to_script("unzip -o /workspace/archive.zip -d /workspace")
        execute_script(script, fs)

        # Should overwrite
        assert fs.read("/workspace/existing.txt") == b"new content"

    def test_unzip_path_traversal_blocked(self, fs):
        """Test that path traversal is blocked."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("../../../etc/passwd", b"evil")
        fs.write("/workspace/evil.zip", buffer.getvalue())

        script = to_script("unzip /workspace/evil.zip -d /workspace")
        with pytest.raises(TerminalError, match="path traversal"):
            execute_script(script, fs)

    def test_unzip_invalid_zip(self, fs):
        """Test error for invalid zip file."""
        fs.write("/workspace/notazip.zip", b"this is not a zip file")

        script = to_script("unzip /workspace/notazip.zip")
        with pytest.raises(TerminalError, match="not a valid zip"):
            execute_script(script, fs)

    def test_unzip_file_not_found(self, fs):
        """Test error for missing zip file."""
        script = to_script("unzip /workspace/nonexistent.zip")
        with pytest.raises(TerminalError, match="cannot find"):
            execute_script(script, fs)


class TestArchiveIntegration:
    """Integration tests combining archive commands."""

    def test_tar_gzip_roundtrip(self, fs):
        """Test creating and extracting tar.gz archive."""
        # Create files
        fs.makedirs("/workspace/src")
        fs.write("/workspace/src/main.py", b"print('hello')")
        fs.write("/workspace/src/utils.py", b"def helper(): pass")

        # Create tar.gz
        script = to_script("tar -czf /workspace/backup.tar.gz /workspace/src")
        execute_script(script, fs)

        # Remove originals
        fs.remove("/workspace/src/main.py")
        fs.remove("/workspace/src/utils.py")

        # Extract
        script = to_script("tar -xzf /workspace/backup.tar.gz -C /workspace/restored")
        execute_script(script, fs)

        # Verify
        assert fs.exists("/workspace/restored/workspace/src/main.py")
        assert fs.read("/workspace/restored/workspace/src/main.py") == b"print('hello')"

    def test_zip_unzip_roundtrip(self, fs):
        """Test creating and extracting zip archive."""
        # Create files
        fs.makedirs("/workspace/project")
        fs.write("/workspace/project/app.py", b"app code")
        fs.write("/workspace/project/config.json", b'{"key": "value"}')

        # Create zip
        script = to_script("zip -r /workspace/project.zip /workspace/project")
        execute_script(script, fs)

        # Remove originals
        fs.remove("/workspace/project/app.py")
        fs.remove("/workspace/project/config.json")

        # Extract
        script = to_script("unzip -o /workspace/project.zip -d /workspace/extracted")
        execute_script(script, fs)

        # Verify
        assert fs.exists("/workspace/extracted/workspace/project/app.py")
        assert fs.read("/workspace/extracted/workspace/project/app.py") == b"app code"
