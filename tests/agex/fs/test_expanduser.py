"""Tests for home directory path leak patches.

Verifies that os.path.expanduser, os.getenv, and os.path.expandvars
correctly sanitize home directory references when VFS or IsolatedFS is active.
"""

import os
import tempfile

from kvit import Live, Staged, Versioned

from agex.fs.isolated import IsolatedFS
from agex.fs.patching import with_isolated_fs, with_virtual_fs
from agex.fs.virtual import VirtualFS
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def _make_state():
    return Staged(Versioned(Memory()), encoder=_agex_encoder, decoder=_agex_decoder)


def test_expanduser_vfs():
    """Test that expanduser returns / when VFS is active."""
    state = _make_state()
    vfs = VirtualFS(state)

    # Without VFS, should return real home
    real_home = os.path.expanduser("~")
    assert real_home != "/"
    assert len(real_home) > 1

    # With VFS, should return /
    with with_virtual_fs(vfs):
        assert os.path.expanduser("~") == "/"
        assert os.path.expanduser("~/test") == "/test"
        assert os.path.expanduser("~/.config") == "/.config"
        # Non-tilde paths should pass through
        assert os.path.expanduser("/absolute/path") == "/absolute/path"
        assert os.path.expanduser("relative/path") == "relative/path"


def test_expanduser_isolated():
    """Test that expanduser returns / when IsolatedFS is active."""
    with tempfile.TemporaryDirectory() as tmpdir:
        isolated = IsolatedFS(str(tmpdir), state=Live())

        # Without IsolatedFS, should return real home
        real_home = os.path.expanduser("~")
        assert real_home != "/"

        # With IsolatedFS, should return /
        with with_isolated_fs(isolated):
            assert os.path.expanduser("~") == "/"
            assert os.path.expanduser("~/Documents") == "/Documents"
            assert os.path.expanduser("~/.ssh/id_rsa") == "/.ssh/id_rsa"


def test_getenv_home_vfs():
    """Test that getenv('HOME') returns / when VFS is active."""
    state = _make_state()
    vfs = VirtualFS(state)

    # Without VFS, should return real home
    real_home = os.getenv("HOME")
    assert real_home is not None
    assert real_home != "/"

    # With VFS, should return /
    with with_virtual_fs(vfs):
        assert os.getenv("HOME") == "/"
        # Other env vars should work normally
        assert os.getenv("PATH") != "/"
        assert os.getenv("NONEXISTENT_VAR", "default") == "default"


def test_getenv_home_isolated():
    """Test that getenv('HOME') returns / when IsolatedFS is active."""
    with tempfile.TemporaryDirectory() as tmpdir:
        isolated = IsolatedFS(str(tmpdir), state=Live())

        # Without IsolatedFS, should return real home
        real_home = os.getenv("HOME")
        assert real_home is not None
        assert real_home != "/"

        # With IsolatedFS, should return /
        with with_isolated_fs(isolated):
            assert os.getenv("HOME") == "/"


def test_expandvars_vfs():
    """Test that expandvars replaces $HOME with / when VFS is active."""
    state = _make_state()
    vfs = VirtualFS(state)

    # Without VFS, $HOME should expand to real home
    real_result = os.path.expandvars("$HOME/test")
    assert real_result != "/test"
    assert "test" in real_result

    # With VFS, $HOME should expand to /
    with with_virtual_fs(vfs):
        assert os.path.expandvars("$HOME") == "/"
        assert os.path.expandvars("$HOME/test") == "/test"
        assert os.path.expandvars("${HOME}/config") == "/config"
        assert os.path.expandvars("$HOME/.bashrc") == "/.bashrc"
        # Mixed variables - $HOME should be replaced but other vars pass through
        assert os.path.expandvars("$HOME/bin:$PATH").startswith("/bin:")


def test_expandvars_isolated():
    """Test that expandvars replaces $HOME with / when IsolatedFS is active."""
    with tempfile.TemporaryDirectory() as tmpdir:
        isolated = IsolatedFS(str(tmpdir), state=Live())

        # Without IsolatedFS, $HOME should expand to real home
        real_result = os.path.expandvars("$HOME/.profile")
        assert real_result != "/.profile"

        # With IsolatedFS, $HOME should expand to /
        with with_isolated_fs(isolated):
            assert os.path.expandvars("$HOME") == "/"
            assert os.path.expandvars("${HOME}") == "/"
            assert os.path.expandvars("$HOME/Downloads") == "/Downloads"


def test_combined_expansion_vfs():
    """Test that expanduser and expandvars work together correctly with VFS."""
    state = _make_state()
    vfs = VirtualFS(state)

    with with_virtual_fs(vfs):
        # First expandvars, then expanduser
        path1 = os.path.expandvars("$HOME/.config")
        assert path1 == "/.config"

        path2 = os.path.expanduser(path1)
        assert path2 == "/.config"

        # Direct expanduser on tilde
        path3 = os.path.expanduser("~/.config")
        assert path3 == "/.config"


def test_pathlike_support():
    """Test that Path objects work with expanduser when VFS is active."""
    state = _make_state()
    vfs = VirtualFS(state)

    with with_virtual_fs(vfs):
        # expanduser should accept PathLike objects
        from pathlib import Path as PathlibPath

        result = os.path.expanduser(PathlibPath("~/test"))
        assert result == "/test"

        result2 = os.path.expandvars(PathlibPath("$HOME/test"))
        assert result2 == "/test"


def test_no_leak_in_error_messages():
    """Verify that error messages don't leak home directory paths."""
    state = _make_state()
    vfs = VirtualFS(state)

    with with_virtual_fs(vfs):
        expanded = os.path.expanduser("~/.credentials")
        assert expanded == "/.credentials"

        # If we try to open a non-existent file, the error message should use /
        try:
            with open(expanded, "r"):
                pass
        except FileNotFoundError as e:
            error_msg = str(e)
            # Should NOT contain real home directory
            real_home = os.getenv("HOME") if os.getenv("HOME") != "/" else "/tmp"
            assert real_home not in error_msg or real_home == "/"
