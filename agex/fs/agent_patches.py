"""Agent-specific filesystem context management.

Contains with_fs_context() which wraps monkeyfs's patch() with
agex-specific concerns (AgentAwareFS unwrapping, commit deferral).
"""

from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

from monkeyfs import defer_commits, patch


@contextmanager
def with_fs_context(
    fs: Any,
    defer: bool = True,
) -> Iterator[None]:
    """Set FS for current async context.

    This is the unified entry point for filesystem context management.

    Args:
        fs: Any FileSystem Protocol implementation (VirtualFS, IsolatedFS,
            AgentAwareFS, or custom).
        defer: If True (default), suppress per-mutation commits to the
            backing store. Set to False to commit on each write.

    Yields:
        None. File operations within the block will use the filesystem.
    """
    from agex.fs.aware import AgentAwareFS

    # Unwrap AgentAwareFS to get the underlying filesystem
    actual_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs

    with ExitStack() as stack:
        if defer:
            stack.enter_context(defer_commits())
        stack.enter_context(patch(actual_fs))
        yield
