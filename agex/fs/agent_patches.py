"""Agent-specific filesystem context management.

Contains with_fs_context() which wraps monkeyfs's patch() with
agex-specific concerns (AgentAwareFS unwrapping).
"""

from contextlib import contextmanager
from typing import Any, Iterator

from monkeyfs import patch


@contextmanager
def with_fs_context(fs: Any) -> Iterator[None]:
    """Set FS for current async context.

    Unwraps AgentAwareFS if needed and activates filesystem interception.

    Args:
        fs: Any FileSystem Protocol implementation (VirtualFS, IsolatedFS,
            AgentAwareFS, or custom).

    Yields:
        None. File operations within the block will use the filesystem.
    """
    from agex.fs.aware import AgentAwareFS

    actual_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs

    with patch(actual_fs):
        yield
