"""Tests for VirtualFS file metadata tracking."""

from gitkv import Live, Staged, Versioned

from agex.fs import VirtualFS
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def _make_state():
    return Staged(Versioned(Memory()), encoder=_agex_encoder, decoder=_agex_decoder)


class TestFileMetadata:
    """Test file metadata tracking (size, timestamps)."""

    def test_write_creates_metadata(self):
        """Test that writing a file creates metadata."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"hello world")

        meta = vfs.stat("file.txt")
        assert meta.size == 11
        assert meta.created_at  # Has timestamp
        assert meta.modified_at  # Has timestamp
        assert meta.created_at == meta.modified_at  # Same for new file

    def test_modify_file_updates_metadata(self):
        """Test that modifying a file updates modified_at but preserves created_at."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"hello")
        original_meta = vfs.stat("file.txt")

        # Modify file
        vfs.write("file.txt", b"hello world again")
        new_meta = vfs.stat("file.txt")

        assert new_meta.size == 17
        assert new_meta.created_at == original_meta.created_at  # Preserved
        assert new_meta.modified_at >= original_meta.modified_at  # Updated

    def test_rename_preserves_created_at(self):
        """Test that renaming preserves created_at timestamp."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("old.txt", b"content")
        original_meta = vfs.stat("old.txt")

        vfs.rename("old.txt", "new.txt")
        new_meta = vfs.stat("new.txt")

        assert new_meta.created_at == original_meta.created_at  # Preserved
        assert new_meta.size == original_meta.size

    def test_remove_deletes_metadata(self):
        """Test that removing a file deletes its metadata."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"content")
        assert vfs.stat("file.txt")  # Metadata exists

        vfs.remove("file.txt")

        # File and metadata gone
        assert not vfs.exists("file.txt")

    def test_stat_fails_for_nonexistent_file(self):
        """Test that stat() raises FileNotFoundError for missing files."""
        state = Live()
        vfs = VirtualFS(state)

        try:
            vfs.stat("nonexistent.txt")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_write_many_creates_metadata_for_all(self):
        """Test that write_many creates metadata for all files."""
        state = Live()
        vfs = VirtualFS(state)

        files = {
            "file1.txt": b"content1",
            "file2.txt": b"content2",
            "dir/file3.txt": b"content3",
        }

        vfs.write_many(files)

        # All files have metadata
        meta1 = vfs.stat("file1.txt")
        assert meta1.size == 8

        meta2 = vfs.stat("file2.txt")
        assert meta2.size == 8

        meta3 = vfs.stat("dir/file3.txt")
        assert meta3.size == 8

    def test_remove_many_deletes_all_metadata(self):
        """Test that remove_many deletes metadata for all files."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write_many({"file1.txt": b"a", "file2.txt": b"b"})
        assert vfs.stat("file1.txt")
        assert vfs.stat("file2.txt")

        vfs.remove_many(["file1.txt", "file2.txt"])

        assert not vfs.exists("file1.txt")
        assert not vfs.exists("file2.txt")

    def test_list_detailed_returns_file_info(self):
        """Test that list_detailed returns FileInfo objects with metadata."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("file1.txt", b"hello")
        vfs.write("file2.txt", b"world")
        vfs.write("dir/file3.txt", b"nested")

        # List root
        files = vfs.list_detailed("/")
        assert len(files) == 3  # file1.txt, file2.txt, dir

        # Find file1.txt
        file1 = next(f for f in files if f.name == "file1.txt")
        assert file1.size == 5
        assert file1.created_at
        assert file1.modified_at
        assert file1.is_dir is False

        # Find dir
        dir_item = next(f for f in files if f.name == "dir")
        assert dir_item.is_dir is True
        assert dir_item.size == 0  # Directories have size 0

    def test_list_detailed_subdirectory(self):
        """Test that list_detailed works for subdirectories."""
        state = Live()
        vfs = VirtualFS(state)

        vfs.write("dir/file1.txt", b"a")
        vfs.write("dir/file2.txt", b"bb")

        files = vfs.list_detailed("/dir")
        assert len(files) == 2

        file1 = next(f for f in files if f.name == "file1.txt")
        assert file1.size == 1
        assert file1.path == "dir/file1.txt"

        file2 = next(f for f in files if f.name == "file2.txt")
        assert file2.size == 2
        assert file2.path == "dir/file2.txt"

    def test_metadata_persists_with_versioned_state(self):
        """Test that metadata persists across snapshots in Versioned state."""
        state = _make_state()
        vfs = VirtualFS(state)

        vfs.write("file.txt", b"content")
        original_meta = vfs.stat("file.txt")

        # Commit
        state.commit()

        # Metadata still accessible
        meta_after = vfs.stat("file.txt")
        assert meta_after.size == original_meta.size
        assert meta_after.created_at == original_meta.created_at
