"""Agent-specific filesystem patches that stay in agex.

Contains with_fs_context() and swap_agent_fs_functions() which depend on
agex internals (AgentAwareFS, agent._policy, etc.).
"""

from contextlib import contextmanager
from typing import Any, Iterator

from monkeyfs.context import vfs_defer_snapshots
from monkeyfs.patching import _vfs_wrappers, install, use_fs


@contextmanager
def with_fs_context(
    fs: Any,
    defer_snapshots: bool = True,
) -> Iterator[None]:
    """Set FS for current async context.

    This is the unified entry point for filesystem context management.

    Args:
        fs: Any FileSystem Protocol implementation (VirtualFS, IsolatedFS,
            AgentAwareFS, or custom).
        defer_snapshots: If True (default), VFS writes won't trigger snapshots.
            Set to False to snapshot on each write (useful for interactive apps).

    Yields:
        None. File operations within the block will use the filesystem.
    """
    from agex.fs.aware import AgentAwareFS

    install()

    # Unwrap AgentAwareFS to get the underlying filesystem
    actual_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs

    # Set defer snapshots flag - when True, VFS writes won't trigger snapshots
    # (prevents recursion with disk-backed state like DiskCache)
    token_defer = vfs_defer_snapshots.set(defer_snapshots)
    try:
        with use_fs(actual_fs):
            yield
    finally:
        vfs_defer_snapshots.reset(token_defer)


def swap_agent_fs_functions(agent: Any) -> None:
    """Swap any registered filesystem functions with FS-aware wrappers.

    This handles the case where agent.fn(open) was called before FS patching,
    or the user explicitly passed the real function reference. Since the
    wrappers check current_fs and fall back to the real implementation
    when no FS is active, it's safe to always use the wrappers.

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

    # Swap registered fs functions with FS-aware wrappers
    fn_objects = main_ns.fn_objects
    for name, func in list(fn_objects.items()):
        if func in _vfs_wrappers:
            fn_objects[name] = _vfs_wrappers[func]

    # Register VirtualFile class so agents can interact with file objects
    if not hasattr(agent, "cls"):
        return

    from monkeyfs import VirtualFile

    # Only register if not already registered
    registered_classes = {rc.cls for rc in main_ns.classes.values()}
    if VirtualFile not in registered_classes:
        agent.cls(VirtualFile, name="VirtualFile")
