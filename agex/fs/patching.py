"""FileSystem patching infrastructure for VirtualFS and IsolatedFS.

Provides context-aware patching of Python's filesystem operations (builtins.open,
os.listdir, etc.) to route to VirtualFS or IsolatedFS when active. Uses contextvars
for async-safe isolation between concurrent agent tasks.

The patching is applied once at module import. Each patched function checks
the context variables to determine whether to use VirtualFS, IsolatedFS, or
the real filesystem.
"""

from __future__ import annotations

import builtins
import errno
import io
import os
import os.path
import pathlib
import site
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from agex.fs.context import current_isolated_fs, current_vfs

if TYPE_CHECKING:
    from agex.fs.base import FileSystem
    from agex.fs.isolated import IsolatedFS
    from agex.fs.virtual import VirtualFS


# Store original implementations once at import
_originals: dict[str, Any] = {
    "open": builtins.open,
    "listdir": os.listdir,
    "remove": os.remove,
    "unlink": os.unlink,
    "mkdir": os.mkdir,
    "makedirs": os.makedirs,
    "rename": os.rename,
    "stat": os.stat,
    "lstat": os.lstat,
    "exists": os.path.exists,
    "isfile": os.path.isfile,
    "isdir": os.path.isdir,
    "getsize": os.path.getsize,
    "scandir": os.scandir,
    "getcwd": os.getcwd,
    "utime": os.utime,
    "touch": Path.touch,
}

# Will be populated after wrapper functions are defined
_vfs_wrappers: dict[Any, Any] = {}


# Define safe system paths for read-only passthrough
# We allow access to stdlib and site-packages even when VFS/IsolatedFS is active
# to support libraries that load their own resources (e.g., plotly, transformers).
def _get_safe_paths() -> list[str]:
    paths = {
        sys.base_prefix,
        sys.prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
    }
    # Add site packages
    for p in site.getsitepackages():
        paths.add(p)
    if hasattr(site, "getusersitepackages"):
        paths.add(site.getusersitepackages())

    # Resolve all paths
    return [str(Path(p).resolve()) for p in paths if os.path.exists(p)]


_SAFE_SYSTEM_PATHS = _get_safe_paths()


# Recursion guard for safe path checks
_in_safe_path_check: ContextVar[bool] = ContextVar("in_safe_path_check", default=False)


def _is_safe_system_path(path: str | Path) -> bool:
    """Check if path is within a safe system directory."""
    try:
        # Prevent recursion when realpath calls lstat/stat
        token = _in_safe_path_check.set(True)
        try:
            # Resolve to absolute path using os.path.realpath
            path_str = os.path.realpath(path)
        finally:
            _in_safe_path_check.reset(token)

        return any(path_str.startswith(sp) for sp in _SAFE_SYSTEM_PATHS)
    except (OSError, ValueError):
        return False


# VFS-aware wrapper functions


def _vfs_open(path: Any, *args: Any, **kwargs: Any) -> Any:
    """FileSystem-aware open() replacement.

    Checks isolated FS first, then virtual FS, then real filesystem.
    """
    # Extract mode if provided, default to "r"
    mode = args[0] if args else kwargs.get("mode", "r")

    # Check isolated FS first
    isolated = current_isolated_fs.get()
    if isolated is not None and isinstance(path, (str, Path)):
        try:
            return isolated.open(str(path), mode, **kwargs)
        except (PermissionError, FileNotFoundError):
            # If read-only and permitted system path, fall through to original
            if (
                "w" not in mode
                and "a" not in mode
                and "+" not in mode
                and "x" not in mode
                and _is_safe_system_path(path)
            ):
                return _originals["open"](path, *args, **kwargs)
            raise

    # Then check virtual FS
    vfs = current_vfs.get()
    if vfs is not None and isinstance(path, (str, Path)):
        try:
            return vfs.open(str(path), mode, **kwargs)
        except FileNotFoundError:
            # If read-only and permitted system path, fall through to original
            if (
                "w" not in mode
                and "a" not in mode
                and "+" not in mode
                and "x" not in mode
                and _is_safe_system_path(path)
            ):
                return _originals["open"](path, *args, **kwargs)
            raise

    return _originals["open"](path, *args, **kwargs)


def _vfs_listdir(path: str = ".") -> list[str]:
    """FileSystem-aware os.listdir() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        try:
            return isolated.listdir(path)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            if _is_safe_system_path(path):
                return _originals["listdir"](path)
            raise

    vfs = current_vfs.get()
    if vfs is not None:
        path_str = str(path)
        # Check if directory exists in VFS
        if vfs.isdir(path_str):
            return vfs.list(path_str)

        # If not in VFS, check if it's a safe system path
        if _is_safe_system_path(path) and _originals["isdir"](path):
            return _originals["listdir"](path)

        # Otherwise raise FileNotFoundError if VFS thought it was missing
        # (vfs.list returns empty list for "any prefix", but real listdir errors if valid dir not found)
        # However, VFS.isdir returned False, so we should error unless safe path.
        raise FileNotFoundError(
            errno.ENOENT, f"No such file or directory: '{path}'", path
        )

    return _originals["listdir"](path)


class MockDirEntry:
    """Mock os.DirEntry for VFS items."""

    def __init__(
        self,
        name: str,
        is_dir: bool,
        stat_result: os.stat_result | None = None,
        path: str | None = None,
    ):
        self.name = name
        self.path = path if path is not None else name
        self._is_dir = is_dir
        self._stat = stat_result

    def is_dir(self, follow_symlinks: bool = True) -> bool:
        return self._is_dir

    def is_file(self, follow_symlinks: bool = True) -> bool:
        return not self._is_dir

    def is_symlink(self) -> bool:
        return False

    def stat(self, follow_symlinks: bool = True) -> os.stat_result:
        if self._stat is None:
            # Fallback if no stat provided (shouldn't happen with our usage)
            raise FileNotFoundError(f"No stat available for {self.name}")
        return self._stat

    def inode(self) -> int:
        return 0

    def __str__(self) -> str:
        return self.path

    def __fspath__(self) -> str:
        return self.path

    def __bytes__(self) -> bytes:
        return os.fsencode(self.path)

    def __repr__(self) -> str:
        return f"<MockDirEntry '{self.name}'>"


class ScandirWrapper:
    """Wrapper to make generator compatible with os.scandir context manager protocol."""

    def __init__(self, iterator: Iterator[os.DirEntry[str]]):
        self._iterator = iterator

    def __iter__(self) -> Iterator[os.DirEntry[str]]:
        return self._iterator

    def __enter__(self) -> "ScandirWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _vfs_scandir(path: str = ".") -> Any:
    """FileSystem-aware os.scandir() replacement."""

    def _scan_gen() -> Iterator[os.DirEntry[str]]:
        import os
        import stat

        # Handle IsolatedFS
        isolated = current_isolated_fs.get()
        if isolated is not None:
            try:
                # Use listdir to get names, then yield MockDirEntries
                resolved = isolated._validate_path(path)

                # Check resolved path safely using original stat
                try:
                    root_st = _originals["stat"](str(resolved))
                except OSError:
                    # If we can't stat the resolved path, it doesn't exist or we can't access it
                    raise NotADirectoryError(f"Not a directory: {path}")

                if not stat.S_ISDIR(root_st.st_mode):
                    raise NotADirectoryError(f"Not a directory: {path}")

                names = isolated.listdir(path)

                for name in names:
                    full_path = str(resolved / name)
                    try:
                        # Use original stat to avoid recursion
                        st = _originals["stat"](full_path)
                        is_d = stat.S_ISDIR(st.st_mode)
                        # Construct entry path relative to the input path to match os.scandir behavior.
                        entry_path = os.path.join(path, name)
                        yield MockDirEntry(name, is_d, st, path=entry_path)  # type: ignore[misc]
                    except OSError:
                        # Skip files that disappeared
                        continue
                return

            except (PermissionError, FileNotFoundError, NotADirectoryError):
                if _is_safe_system_path(path):
                    # We need to yield from the original scandir
                    with _originals["scandir"](path) as it:
                        yield from it
                    return
                raise

        # Handle VirtualFS
        vfs = current_vfs.get()
        if vfs is not None:
            path_str = str(path)

            # Check if directory exists in VFS
            if vfs.isdir(path_str):
                from agex.fs.patching import _vfs_stat  # Use our patched stat helper

                names = vfs.list(path_str)
                for name in names:
                    # Construct child path for stat
                    child_path = os.path.join(path_str, name)

                    try:
                        # Get stat (using _vfs_stat which handles VFS logic)
                        st = _vfs_stat(child_path)
                        is_d = vfs.isdir(child_path)
                        yield MockDirEntry(name, is_d, st, path=child_path)  # type: ignore[misc]
                    except FileNotFoundError:
                        continue
                return

            # If not in VFS, check if it's a safe system path
            if _is_safe_system_path(path) and _originals["isdir"](path):
                with _originals["scandir"](path) as it:
                    yield from it
                return

            raise FileNotFoundError(
                errno.ENOENT, f"No such file or directory: '{path}'", path
            )

        # No VFS/IsolatedFS active
        with _originals["scandir"](path) as it:
            yield from it

    return ScandirWrapper(_scan_gen())


def _vfs_remove(path: str, **kwargs: Any) -> None:
    """FileSystem-aware os.remove() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        return isolated.remove(path)

    vfs = current_vfs.get()
    if vfs is not None:
        return vfs.remove(str(path), snapshot=False)

    return _originals["remove"](path, **kwargs)


def _vfs_unlink(path: str, **kwargs: Any) -> None:
    """FileSystem-aware os.unlink() replacement (alias for remove)."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        return isolated.remove(path)

    vfs = current_vfs.get()
    if vfs is not None:
        return vfs.remove(str(path), snapshot=False)

    return _originals["unlink"](path, **kwargs)


def _vfs_mkdir(path: str, mode: int = 0o777, **kwargs: Any) -> None:
    """FileSystem-aware os.mkdir() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        return isolated.mkdir(path)

    vfs = current_vfs.get()
    if vfs is not None:
        return vfs.mkdir(str(path))

    return _originals["mkdir"](path, mode, **kwargs)


def _vfs_makedirs(path: str, mode: int = 0o777, exist_ok: bool = False) -> None:
    """FileSystem-aware os.makedirs() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        return isolated.mkdir(path, parents=True, exist_ok=exist_ok)

    vfs = current_vfs.get()
    if vfs is not None:
        return vfs.makedirs(str(path), exist_ok=exist_ok)

    return _originals["makedirs"](path, mode, exist_ok=exist_ok)


def _vfs_rename(src: str, dst: str, **kwargs: Any) -> None:
    """FileSystem-aware os.rename() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        return isolated.rename(src, dst)

    vfs = current_vfs.get()
    if vfs is not None:
        return vfs.rename(str(src), str(dst), snapshot=False)

    return _originals["rename"](src, dst, **kwargs)


def _vfs_stat(path: str, **kwargs: Any) -> Any:
    """FileSystem-aware os.stat() replacement.

    Returns stat_result with metadata from filesystem when active.
    """
    # Break recursion from realpath -> lstat -> _vfs_stat
    if _in_safe_path_check.get():
        return _originals["stat"](path, **kwargs)

    # Check isolated FS first - it uses real stat
    isolated = current_isolated_fs.get()
    if isolated is not None:
        try:
            # For IsolatedFS, we want the real stat result (for pathlib compatibility)
            # incorrectly usage of internal API but necessary for full stat support
            resolved = isolated._validate_path(path)
            return _originals["stat"](str(resolved), **kwargs)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            if _is_safe_system_path(path):
                return _originals["stat"](path, **kwargs)
            raise

    # Then check virtual FS
    vfs = current_vfs.get()
    if vfs is not None:
        import stat as stat_module
        from datetime import datetime

        # Convert pathlib.Path to string if needed (pandas may pass Path objects)
        path_str = str(path)

        # Check if it's a file
        if vfs.isfile(path_str):
            metadata = vfs.stat(path_str)

            # Parse ISO timestamps to epoch floats
            try:
                created_ts = datetime.fromisoformat(metadata.created_at).timestamp()
                modified_ts = datetime.fromisoformat(metadata.modified_at).timestamp()
            except (ValueError, AttributeError):
                # Fallback to current time if parsing fails
                import time

                created_ts = modified_ts = time.time()

            # Construct stat_result for file
            # Mock values for permissions (0o644), UID/GID (1000), device info
            return os.stat_result(
                (
                    stat_module.S_IFREG | 0o644,  # st_mode: regular file, rw-r--r--
                    0,  # st_ino: inode (not meaningful in VFS)
                    0,  # st_dev: device (not meaningful in VFS)
                    1,  # st_nlink: number of hard links
                    1000,  # st_uid: user ID (mocked)
                    1000,  # st_gid: group ID (mocked)
                    metadata.size,  # st_size: file size in bytes
                    modified_ts,  # st_atime: access time
                    modified_ts,  # st_mtime: modification time
                    created_ts,  # st_ctime: creation time
                )
            )

        # Check if it's a directory
        elif vfs.isdir(path_str):
            import time

            current_time = time.time()

            # Construct stat_result for directory
            return os.stat_result(
                (
                    stat_module.S_IFDIR | 0o755,  # st_mode: directory, rwxr-xr-x
                    0,  # st_ino
                    0,  # st_dev
                    2,  # st_nlink: directories typically have 2+
                    1000,  # st_uid
                    1000,  # st_gid
                    0,  # st_size: directories are zero
                    current_time,  # st_atime
                    current_time,  # st_mtime
                    current_time,  # st_ctime
                )
            )

        # Path doesn't exist in VFS
        else:
            # Check safe paths before raising
            if _is_safe_system_path(path):
                return _originals["stat"](path, **kwargs)

            raise FileNotFoundError(
                errno.ENOENT, f"No such file or directory: '{path}'", path
            )

    # Fallback to original stat
    return _originals["stat"](path, **kwargs)


def _vfs_lstat(path: str, **kwargs: Any) -> Any:
    """FileSystem-aware os.lstat() replacement."""
    # Break recursion from realpath -> lstat -> _vfs_lstat
    if _in_safe_path_check.get():
        return _originals["lstat"](path, **kwargs)

    # Check isolated FS first - it uses real lstat
    isolated = current_isolated_fs.get()
    if isolated is not None:
        try:
            # For IsolatedFS, we want the real lstat result
            resolved = isolated._validate_path(path)
            # Use lstat explicitly to preserve symlink information
            return _originals["lstat"](str(resolved), **kwargs)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            if _is_safe_system_path(path):
                return _originals["lstat"](path, **kwargs)
            raise

    # Then check virtual FS (same as stat since VFS has no symlinks yet)
    vfs = current_vfs.get()
    if vfs is not None:
        return _vfs_stat(path, **kwargs)

    # Fallback to original lstat
    return _originals["lstat"](path, **kwargs)


def _vfs_exists(path: str, **kwargs: Any) -> bool:
    """FileSystem-aware os.path.exists() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        try:
            return isolated.exists(path)
        except PermissionError:
            if _is_safe_system_path(path):
                return _originals["exists"](path, **kwargs)
            return False

    vfs = current_vfs.get()
    if vfs is not None:
        path_str = str(path)
        if vfs.exists(path_str):
            return True
        if _is_safe_system_path(path):
            return _originals["exists"](path, **kwargs)
        return False

    return _originals["exists"](path, **kwargs)


def _vfs_isfile(path: str, **kwargs: Any) -> bool:
    """FileSystem-aware os.path.isfile() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        try:
            return isolated.isfile(path)
        except PermissionError:
            if _is_safe_system_path(path):
                return _originals["isfile"](path, **kwargs)
            return False

    vfs = current_vfs.get()
    if vfs is not None:
        path_str = str(path)
        if vfs.isfile(path_str):
            return True
        if _is_safe_system_path(path):
            return _originals["isfile"](path, **kwargs)
        return False

    return _originals["isfile"](path, **kwargs)


def _vfs_isdir(path: str, **kwargs: Any) -> bool:
    """FileSystem-aware os.path.isdir() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        try:
            return isolated.isdir(path)
        except PermissionError:
            if _is_safe_system_path(path):
                return _originals["isdir"](path, **kwargs)
            return False

    vfs = current_vfs.get()
    if vfs is not None:
        path_str = str(path)
        if vfs.isdir(path_str):
            return True
        if _is_safe_system_path(path):
            return _originals["isdir"](path, **kwargs)
        return False

    return _originals["isdir"](path, **kwargs)


def _vfs_getsize(path: str, **kwargs: Any) -> int:
    """FileSystem-aware os.path.getsize() replacement."""
    isolated = current_isolated_fs.get()
    if isolated is not None:
        try:
            return isolated.stat(path).size
        except (PermissionError, FileNotFoundError):
            if _is_safe_system_path(path):
                return _originals["getsize"](path, **kwargs)
            raise

    vfs = current_vfs.get()
    if vfs is not None:
        try:
            return vfs.getsize(str(path))
        except FileNotFoundError:
            if _is_safe_system_path(path):
                return _originals["getsize"](path, **kwargs)
            raise

    return _originals["getsize"](path, **kwargs)


def _vfs_getcwd() -> str:
    """FileSystem-aware os.getcwd() replacement.

    Returns '/' if a virtual or isolated filesystem is active, otherwise
    returns the real current working directory.
    """
    if current_isolated_fs.get() is not None:
        return "/"

    if current_vfs.get() is not None:
        return "/"

    return _originals["getcwd"]()


def _vfs_utime(
    path: str | bytes | os.PathLike[Any],
    times: tuple[int, int] | tuple[float, float] | None = None,
    **kwargs: Any,
) -> None:
    """FileSystem-aware os.utime() replacement."""
    # We don't currently support VFS timestamp updates, but we want to avoid
    # implementation-dependent errors when tools try to touch files.
    # For now, pass through to isolated (which supports it) or swallow for VFS
    # if the file exists.

    path_str = str(path)

    isolated = current_isolated_fs.get()
    if isolated is not None:
        try:
            return _originals["utime"](
                str(isolated._validate_path(path_str)), times, **kwargs
            )
        except (PermissionError, FileNotFoundError):
            if _is_safe_system_path(path_str):
                return _originals["utime"](path_str, times, **kwargs)
            raise

    vfs = current_vfs.get()
    if vfs is not None:
        if vfs.exists(path_str):
            # TODO: Implement metadata updates in VFS
            # For now, just silently succeed to allow 'touch' to work
            return

        if _is_safe_system_path(path_str):
            return _originals["utime"](path_str, times, **kwargs)

        raise FileNotFoundError(
            errno.ENOENT, f"No such file or directory: '{path_str}'", path_str
        )

    return _originals["utime"](path, times, **kwargs)


def _vfs_touch(self: Path, mode: int = 0o666, exist_ok: bool = True) -> None:
    """FileSystem-aware pathlib.Path.touch() replacement.

    This ensures Path.touch() uses the patched open() instead of OS-level open.
    """
    if exist_ok:
        # First try to open for append (to avoid truncation) to check existence/create
        # But touch semantics are tricky. Simple touch:
        # 1. If exists, update times (utime)
        # 2. If not, create empty file
        try:
            # Check existence first to decide whether to update timestamps or create
            if self.exists():
                os.utime(self, None)
                return
        except FileNotFoundError:
            pass  # Does not exist, proceed to create

        # Create empty file using patched open (append mode safe/create)
        with builtins.open(self, "a"):
            pass
    else:
        # exist_ok=False: Expect exclusive creation, raise FileExistsError if exists
        with builtins.open(self, "x"):
            pass


def apply_patches() -> None:
    """Apply VFS-aware patches to builtins and os module.

    This should be called once at module import time. The patches are
    permanent but only affect behavior when _current_vfs is set.
    """
    # Patch builtins
    builtins.open = _vfs_open  # type: ignore[assignment]
    io.open = _vfs_open  # type: ignore[assignment]

    # Patch os module
    os.listdir = _vfs_listdir  # type: ignore[assignment]
    os.remove = _vfs_remove  # type: ignore[assignment]
    os.unlink = _vfs_unlink  # type: ignore[assignment]
    os.mkdir = _vfs_mkdir  # type: ignore[assignment]
    os.makedirs = _vfs_makedirs  # type: ignore[assignment]
    os.rename = _vfs_rename  # type: ignore[assignment]
    os.stat = _vfs_stat  # type: ignore[assignment]
    os.lstat = _vfs_lstat  # type: ignore[assignment]
    os.scandir = _vfs_scandir  # type: ignore[assignment]
    os.getcwd = _vfs_getcwd  # type: ignore[assignment]
    os.utime = _vfs_utime  # type: ignore[assignment]

    # Patch pathlib.Path.touch
    Path.touch = _vfs_touch  # type: ignore[assignment]

    # Patch os.path
    os.path.exists = _vfs_exists  # type: ignore[assignment]
    os.path.isfile = _vfs_isfile  # type: ignore[assignment]
    os.path.isdir = _vfs_isdir  # type: ignore[assignment]
    os.path.getsize = _vfs_getsize  # type: ignore[assignment]

    # Copy metadata from originals to wrappers so agent.fn(open) registers as 'open'
    _vfs_open.__name__ = "open"
    _vfs_open.__doc__ = _originals["open"].__doc__
    _vfs_listdir.__name__ = "listdir"
    _vfs_remove.__name__ = "remove"
    _vfs_unlink.__name__ = "unlink"
    _vfs_mkdir.__name__ = "mkdir"
    _vfs_makedirs.__name__ = "makedirs"
    _vfs_rename.__name__ = "rename"
    _vfs_stat.__name__ = "stat"
    _vfs_scandir.__name__ = "scandir"
    _vfs_getcwd.__name__ = "getcwd"
    _vfs_utime.__name__ = "utime"
    _vfs_exists.__name__ = "exists"
    _vfs_isfile.__name__ = "isfile"
    _vfs_isdir.__name__ = "isdir"
    _vfs_isfile.__name__ = "isfile"
    _vfs_isdir.__name__ = "isdir"
    _vfs_getsize.__name__ = "getsize"

    # Patch pathlib internal accessor (Python < 3.11, e.g. 3.10)
    # Older pathlib implementations cache os functions in _NormalAccessor
    # So we need to patch those cached references directly.
    if hasattr(pathlib, "_NormalAccessor"):
        accessor = pathlib._NormalAccessor  # type: ignore
        if hasattr(accessor, "stat"):
            accessor.stat = staticmethod(_vfs_stat)  # staticmethod on class
        if hasattr(accessor, "lstat"):
            accessor.lstat = staticmethod(_vfs_lstat)
        if hasattr(accessor, "scandir"):
            accessor.scandir = staticmethod(_vfs_scandir)
        if hasattr(accessor, "open"):
            accessor.open = staticmethod(_vfs_open)
        if hasattr(accessor, "unlink"):
            accessor.unlink = staticmethod(_vfs_unlink)
        if hasattr(accessor, "rmdir"):
            accessor.rmdir = staticmethod(
                _vfs_remove
            )  # rmdir/remove often same or similar enough
        if hasattr(accessor, "rename"):
            accessor.rename = staticmethod(_vfs_rename)
        if hasattr(accessor, "mkdir"):
            accessor.mkdir = staticmethod(_vfs_mkdir)
        if hasattr(accessor, "listdir"):
            accessor.listdir = staticmethod(_vfs_listdir)
        if hasattr(accessor, "getcwd"):
            accessor.getcwd = staticmethod(_vfs_getcwd)

    # Patch pathlib.Path._globber (Python 3.13+)
    # It captures os.scandir/lstat at definition time
    if hasattr(pathlib.Path, "_globber"):
        globber = pathlib.Path._globber  # type: ignore
        if hasattr(globber, "scandir"):
            globber.scandir = staticmethod(_vfs_scandir)
        if hasattr(globber, "lstat"):
            globber.lstat = staticmethod(
                _vfs_lstat
            )  # _vfs_stat handles VFS and delegates safely


@contextmanager
def with_virtual_fs(vfs: "VirtualFS") -> Iterator[None]:
    """Set VFS for current async context.

    This context manager sets the virtual filesystem for the duration of
    the with block. It is async-safe - concurrent async tasks each get
    their own context.

    Args:
        vfs: VirtualFS instance to use.

    Yields:
        None. File operations within the block will use the VFS.

    Example:
        >>> with with_virtual_fs(vfs):
        ...     with open("data.csv", "w") as f:
        ...         f.write("a,b,c")
        ...     # File is written to VFS, not real filesystem
    """
    token = current_vfs.set(vfs)
    try:
        yield
    finally:
        current_vfs.reset(token)


@contextmanager
def with_isolated_fs(isolated_fs: "IsolatedFS") -> Iterator[None]:
    """Set isolated FS for current async context.

    This context manager sets the isolated filesystem for the duration of
    the with block. It is async-safe - concurrent async tasks each get
    their own context.

    Args:
        isolated_fs: IsolatedFS instance to use.

    Yields:
        None. File operations within the block will use the isolated FS.

    Example:
        >>> from agex.fs import IsolatedFS
        >>> isolated = IsolatedFS(root="/path/to/project")
        >>> with with_isolated_fs(isolated):
        ...     with open("data.csv", "w") as f:
        ...         f.write("a,b,c")
        ...     # File is written to real filesystem within root
    """
    token = current_isolated_fs.set(isolated_fs)
    try:
        yield
    finally:
        current_isolated_fs.reset(token)


@contextmanager
def with_fs_context(fs: "FileSystem") -> Iterator[None]:
    """Set FS for current async context based on filesystem type.

    Dispatches to the appropriate context manager based on filesystem type.
    This is the unified entry point for filesystem context management.

    Args:
        fs: FileSystem instance (VirtualFS, IsolatedFS, or AgentAwareFS) to use.

    Yields:
        None. File operations within the block will use the filesystem.

    Example:
        >>> with with_fs_context(fs):
        ...     with open("data.csv", "w") as f:
        ...         f.write("a,b,c")
        ...     # File is written to the appropriate filesystem
    """
    from agex.fs.aware import AgentAwareFS
    from agex.fs.isolated import IsolatedFS
    from agex.fs.virtual import VirtualFS

    # Unwrap AgentAwareFS to get the underlying filesystem
    actual_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs

    if isinstance(actual_fs, VirtualFS):
        with with_virtual_fs(actual_fs):
            yield
    elif isinstance(actual_fs, IsolatedFS):
        with with_isolated_fs(actual_fs):
            yield
    else:
        raise TypeError(f"Unknown filesystem type: {type(actual_fs)}")


def get_current_vfs() -> "VirtualFS | None":
    """Get the current VFS for the async context.

    Returns:
        The current VirtualFS, or None if not in a VFS context.
    """
    return current_vfs.get()


def get_current_isolated_fs() -> "IsolatedFS | None":
    """Get the current isolated FS for the async context.

    Returns:
        The current IsolatedFS, or None if not in an isolated FS context.
    """
    return current_isolated_fs.get()


def swap_agent_fs_functions(agent: Any) -> None:
    """Swap any registered filesystem functions with VFS-aware wrappers.

    This handles the case where agent.fn(open) was called before VFS patching,
    or the user explicitly passed the real function reference. Since the
    VFS wrappers check _current_vfs and fall back to the real implementation
    when VFS isn't active, it's safe to always use the wrappers.

    Also registers StringIO and BytesIO so agent code can call methods like
    .read() on file objects returned by open().

    This is a one-time swap and doesn't need to be reversed.

    Args:
        agent: The agent whose registered functions should be swapped.
    """

    if not hasattr(agent, "_policy"):
        return

    # Ensure __main__ namespace exists (creates if missing)
    main_ns = agent._policy._get_or_create_main()

    # Ensure IO modules are available (late import to avoid cycles)
    from agex.helpers.stdlib import register_io

    register_io(agent)

    # Swap registered fs functions with VFS-aware wrappers
    fn_objects = main_ns.fn_objects
    for name, func in list(fn_objects.items()):
        if func in _vfs_wrappers:
            fn_objects[name] = _vfs_wrappers[func]

    # Register VirtualFile class so agents can interact with file objects
    if not hasattr(agent, "cls"):
        return

    # Import VirtualFile locally to avoid circular imports
    from agex.fs.virtual import VirtualFile

    # Only register if not already registered
    registered_classes = {rc.cls for rc in main_ns.classes.values()}
    if VirtualFile not in registered_classes:
        agent.cls(VirtualFile, name="VirtualFile")


# Apply patches at module import
apply_patches()

# Populate wrapper mapping AFTER wrappers are defined
# Maps original function -> VFS-aware wrapper
_vfs_wrappers.update(
    {
        _originals["open"]: _vfs_open,
        _originals["listdir"]: _vfs_listdir,
        _originals["remove"]: _vfs_remove,
        _originals["unlink"]: _vfs_unlink,
        _originals["mkdir"]: _vfs_mkdir,
        _originals["makedirs"]: _vfs_makedirs,
        _originals["rename"]: _vfs_rename,
        _originals["stat"]: _vfs_stat,
        _originals["scandir"]: _vfs_scandir,
        _originals["exists"]: _vfs_exists,
        _originals["isfile"]: _vfs_isfile,
        _originals["isdir"]: _vfs_isdir,
        _originals["getsize"]: _vfs_getsize,
    }
)
