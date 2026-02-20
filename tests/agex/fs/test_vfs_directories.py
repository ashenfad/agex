import pytest
from kvit import Live

from agex.fs.virtual import VirtualFS


def test_vfs_directory_behavior():
    """Verify that VirtualFS handles implicit directories correctly."""
    state = Live()
    vfs = VirtualFS(state)

    # Write a nested file
    vfs.write("a/b/c.py", b"print('hello')")

    # 1. Verify existence of parents
    assert vfs.exists("a/b/c.py")
    assert vfs.exists("a/b")
    assert vfs.exists("a")
    assert vfs.exists(".")
    assert vfs.exists("/")

    # 2. Verify isdir
    assert vfs.isdir("a")
    assert vfs.isdir("a/b")
    assert not vfs.isdir("a/b/c.py")

    # 3. Verify isfile
    assert vfs.isfile("a/b/c.py")
    assert not vfs.isfile("a/b")
    assert not vfs.isfile("a")

    # 4. Verify list
    assert vfs.list("/") == ["a"]
    assert vfs.list("a") == ["b"]
    assert vfs.list("a/b") == ["c.py"]

    # 5. Verify case sensitivity/normalization
    assert vfs.exists("./a/b/c.py")
    assert vfs.exists("/a/b/c.py")


def test_vfs_list_multiple_items():
    """Verify listing works with multiple items in same directory."""
    state = Live()
    vfs = VirtualFS(state)

    vfs.write("pkg/__init__.py", b"")
    vfs.write("pkg/utils.py", b"")
    vfs.write("pkg/sub/mod.py", b"")
    vfs.write("top.py", b"")

    assert vfs.list("/") == ["pkg", "top.py"]
    assert vfs.list("pkg") == ["__init__.py", "sub", "utils.py"]
    assert vfs.list("pkg/sub") == ["mod.py"]


if __name__ == "__main__":
    pytest.main([__file__])
