"""Filesystem patching infrastructure for VirtualFS and IsolatedFS.

Provides context-aware patching of Python's filesystem operations (builtins.open,
os.listdir, etc.) to route to VirtualFS or IsolatedFS when active. Uses contextvars
for async-safe isolation between concurrent agent tasks.

The patching is applied once at module import. Each patched function checks
the context variables to determine whether to use VirtualFS, IsolatedFS, or
the real filesystem.
"""

from __future__ import annotations

import builtins
import contextvars
import os
import os.path
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from agex.fs.isolated import IsolatedFS
    from agex.fs.virtual import VirtualFS


# Context variables holding the current filesystems
# Isolated FS is checked first, then virtual FS, then real filesystem
_current_isolated_fs: contextvars.ContextVar[IsolatedFS | None] = (
    contextvars.ContextVar("agex_current_isolated_fs", default=None)
)
_current_vfs: contextvars.ContextVar[VirtualFS | None] = contextvars.ContextVar(
    "agex_current_vfs", default=None
)

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
    "exists": os.path.exists,
    "isfile": os.path.isfile,
    "isdir": os.path.isdir,
    "getsize": os.path.getsize,
}

# Will be populated after wrapper functions are defined
_vfs_wrappers: dict[Any, Any] = {}


# VFS-aware wrapper functions


def _vfs_open(path: Any, *args: Any, **kwargs: Any) -> Any:
    """Filesystem-aware open() replacement.

    Checks isolated FS first, then virtual FS, then real filesystem.
    """
    # Extract mode if provided, default to "r"
    mode = args[0] if args else kwargs.get("mode", "r")

    # Check isolated FS first
    isolated = _current_isolated_fs.get()
    if isolated is not None and isinstance(path, str):
        return isolated.open(path, mode, **kwargs)

    # Then check virtual FS
    vfs = _current_vfs.get()
    if vfs is not None and isinstance(path, str):
        return vfs.open(path, mode, **kwargs)

    return _originals["open"](path, *args, **kwargs)


def _vfs_listdir(path: str = ".") -> list[str]:
    """Filesystem-aware os.listdir() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.listdir(path)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.list(path)

    return _originals["listdir"](path)


def _vfs_remove(path: str, **kwargs: Any) -> None:
    """Filesystem-aware os.remove() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.remove(path)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.remove(path, snapshot=False)

    return _originals["remove"](path, **kwargs)


def _vfs_unlink(path: str, **kwargs: Any) -> None:
    """Filesystem-aware os.unlink() replacement (alias for remove)."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.remove(path)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.remove(path, snapshot=False)

    return _originals["unlink"](path, **kwargs)


def _vfs_mkdir(path: str, mode: int = 0o777, **kwargs: Any) -> None:
    """Filesystem-aware os.mkdir() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.mkdir(path)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.mkdir(path)

    return _originals["mkdir"](path, mode, **kwargs)


def _vfs_makedirs(path: str, mode: int = 0o777, exist_ok: bool = False) -> None:
    """Filesystem-aware os.makedirs() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.mkdir(path, parents=True, exist_ok=exist_ok)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.makedirs(path, exist_ok=exist_ok)

    return _originals["makedirs"](path, mode, exist_ok=exist_ok)


def _vfs_rename(src: str, dst: str, **kwargs: Any) -> None:
    """Filesystem-aware os.rename() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.rename(src, dst)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.rename(src, dst, snapshot=False)

    return _originals["rename"](src, dst, **kwargs)


def _vfs_stat(path: str, **kwargs: Any) -> Any:
    """Filesystem-aware os.stat() replacement.

    Returns stat_result with metadata from filesystem when active.
    """
    # Check isolated FS first - it uses real stat
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return _originals["stat"](str(path), **kwargs)

    # Then check virtual FS
    vfs = _current_vfs.get()
    if vfs is not None:
        import stat as stat_module
        from datetime import datetime

        # Convert pathlib.Path to string if needed (pandas may pass Path objects)
        path = str(path)

        # Check if it's a file
        if vfs.isfile(path):
            metadata = vfs.stat(path)

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
        elif vfs.isdir(path):
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

        # Path doesn't exist
        else:
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{path}'")

    return _originals["stat"](path, **kwargs)


def _vfs_exists(path: str, **kwargs: Any) -> bool:
    """Filesystem-aware os.path.exists() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.exists(path)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.exists(path)

    return _originals["exists"](path, **kwargs)


def _vfs_isfile(path: str, **kwargs: Any) -> bool:
    """Filesystem-aware os.path.isfile() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.isfile(path)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.isfile(path)

    return _originals["isfile"](path, **kwargs)


def _vfs_isdir(path: str, **kwargs: Any) -> bool:
    """Filesystem-aware os.path.isdir() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.isdir(path)

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.isdir(path)

    return _originals["isdir"](path, **kwargs)


def _vfs_getsize(path: str, **kwargs: Any) -> int:
    """Filesystem-aware os.path.getsize() replacement."""
    isolated = _current_isolated_fs.get()
    if isolated is not None:
        return isolated.stat(path).size

    vfs = _current_vfs.get()
    if vfs is not None:
        return vfs.getsize(path)

    return _originals["getsize"](path, **kwargs)


def apply_patches() -> None:
    """Apply VFS-aware patches to builtins and os module.

    This should be called once at module import time. The patches are
    permanent but only affect behavior when _current_vfs is set.
    """
    # Patch builtins
    builtins.open = _vfs_open  # type: ignore[assignment]

    # Patch os module
    os.listdir = _vfs_listdir  # type: ignore[assignment]
    os.remove = _vfs_remove  # type: ignore[assignment]
    os.unlink = _vfs_unlink  # type: ignore[assignment]
    os.mkdir = _vfs_mkdir  # type: ignore[assignment]
    os.makedirs = _vfs_makedirs  # type: ignore[assignment]
    os.rename = _vfs_rename  # type: ignore[assignment]
    os.stat = _vfs_stat  # type: ignore[assignment]

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
    _vfs_exists.__name__ = "exists"
    _vfs_isfile.__name__ = "isfile"
    _vfs_isdir.__name__ = "isdir"
    _vfs_getsize.__name__ = "getsize"


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
    token = _current_vfs.set(vfs)
    try:
        yield
    finally:
        _current_vfs.reset(token)


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
    token = _current_isolated_fs.set(isolated_fs)
    try:
        yield
    finally:
        _current_isolated_fs.reset(token)


def get_current_vfs() -> "VirtualFS | None":
    """Get the current VFS for the async context.

    Returns:
        The current VirtualFS, or None if not in a VFS context.
    """
    return _current_vfs.get()


def get_current_isolated_fs() -> "IsolatedFS | None":
    """Get the current isolated FS for the async context.

    Returns:
        The current IsolatedFS, or None if not in an isolated FS context.
    """
    return _current_isolated_fs.get()


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
        _originals["exists"]: _vfs_exists,
        _originals["isfile"]: _vfs_isfile,
        _originals["isdir"]: _vfs_isdir,
        _originals["getsize"]: _vfs_getsize,
    }
)
