import os
import os.path

from agex.fs.patching import with_virtual_fs
from agex.fs.virtual import VirtualFS
from agex.state import Live


def test_vfs_islink():
    state = Live()
    vfs = VirtualFS(state)
    vfs.write("test.txt", b"content")

    with with_virtual_fs(vfs):
        assert os.path.islink("test.txt") is False
        assert os.path.islink("nonexistent.txt") is False


def test_vfs_lexists():
    state = Live()
    vfs = VirtualFS(state)
    vfs.write("test.txt", b"content")

    with with_virtual_fs(vfs):
        assert os.path.lexists("test.txt") is True
        assert os.path.lexists("nonexistent.txt") is False


def test_vfs_samefile():
    state = Live()
    vfs = VirtualFS(state)
    vfs.write("test.txt", b"content")

    with with_virtual_fs(vfs):
        assert os.path.samefile("test.txt", "test.txt") is True
        assert os.path.samefile("test.txt", "./test.txt") is True

        vfs.write("other.txt", b"other")
        assert os.path.samefile("test.txt", "other.txt") is False


def test_vfs_realpath():
    state = Live()
    vfs = VirtualFS(state)
    vfs.write("test.txt", b"content")

    with with_virtual_fs(vfs):
        # realpath should be absolute and normalized
        assert os.path.realpath("test.txt") == "/test.txt"
        assert os.path.realpath("./test.txt") == "/test.txt"
        assert os.path.realpath("/test.txt") == "/test.txt"
        assert os.path.realpath("dir/../test.txt") == "/test.txt"


def test_isolated_realpath(tmp_path):
    from agex.fs.isolated import IsolatedFS
    from agex.fs.patching import with_isolated_fs

    root = tmp_path / "root"
    root.mkdir()
    (root / "test.txt").write_text("content")

    isolated = IsolatedFS(str(root), state=Live())

    with with_isolated_fs(isolated):
        assert os.path.realpath("test.txt") == "/test.txt"
        assert os.path.realpath("./test.txt") == "/test.txt"
        assert os.path.realpath("/test.txt") == "/test.txt"


def test_isolated_islink(tmp_path):
    from agex.fs.isolated import IsolatedFS
    from agex.fs.patching import with_isolated_fs

    root = tmp_path / "root"
    root.mkdir()
    file = root / "test.txt"
    file.write_text("content")

    link = root / "link.txt"
    link.symlink_to(file)

    isolated = IsolatedFS(str(root), state=Live())

    with with_isolated_fs(isolated):
        assert os.path.islink("link.txt") is True
        assert os.path.islink("test.txt") is False


def test_isolated_samefile(tmp_path):
    from agex.fs.isolated import IsolatedFS
    from agex.fs.patching import with_isolated_fs

    root = tmp_path / "root"
    root.mkdir()
    file = root / "test.txt"
    file.write_text("content")

    link = root / "link.txt"
    link.symlink_to(file)

    isolated = IsolatedFS(str(root), state=Live())

    with with_isolated_fs(isolated):
        assert os.path.samefile("test.txt", "link.txt") is True
        assert os.path.samefile("test.txt", "test.txt") is True
