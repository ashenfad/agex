"""FileSystem support for agex agents.

This module provides both virtual (in-memory) and isolated (restricted real)
filesystem access for agents via standard Python file operations.

Core FS functionality is provided by the monkeyfs library. This module
re-exports its public API and adds agex-specific extensions:
    AgentAwareFS: FS wrapper that emits events for agent visibility
    with_fs_context: Unified entry point for filesystem context management
    connect_fs: Wraps monkeyfs connect_fs with agex-specific options (per_session)
"""

from typing import Literal

# Re-export monkeyfs public API
from monkeyfs import (
    FileInfo,
    FileMetadata,
    FileSystem,
    FSConfig,
    IsolatedFS,
    IsolatedFSConfig,
    VirtualFile,
    VirtualFS,
    VirtualFSConfig,
    patch,
    suspend,
)
from monkeyfs import connect_fs as _monkeyfs_connect_fs

# agex-specific
from agex.fs.agent_patches import with_fs_context
from agex.fs.aware import AgentAwareFS


def connect_fs(
    type: Literal["virtual", "isolated"] = "virtual",
    **kwargs,
) -> FSConfig:
    """Configure filesystem access.

    Wraps monkeyfs's connect_fs with agex-specific options.

    Additional kwargs for type="isolated":
        per_session (bool): Create session subdirectories (default: False).
    """
    per_session = kwargs.pop("per_session", False)
    # tracking was removed — accept and ignore for backwards compat
    kwargs.pop("tracking", None)

    config = _monkeyfs_connect_fs(type=type, **kwargs)

    if isinstance(config, IsolatedFSConfig) and per_session:
        config.per_session = per_session  # type: ignore[attr-defined]

    return config


__all__ = [
    "AgentAwareFS",
    "connect_fs",
    "FileInfo",
    "FileMetadata",
    "FileSystem",
    "FSConfig",
    "IsolatedFS",
    "IsolatedFSConfig",
    "suspend",
    "patch",
    "VirtualFile",
    "VirtualFS",
    "VirtualFSConfig",
    "with_fs_context",
]
