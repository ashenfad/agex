"""FileSystem support for agex agents.

This module provides both virtual (in-memory) and isolated (restricted real)
filesystem access for agents via standard Python file operations.

Core FS functionality is provided by the monkeyfs library. This module
re-exports its public API and adds agex-specific extensions:
    AgentAwareFS: FS wrapper that emits events for agent visibility
    with_fs_context: Unified entry point for filesystem context management
"""

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
    connect_fs,
    get_current_fs,
    patch,
    suspend,
)

# agex-specific
from agex.fs.agent_patches import with_fs_context
from agex.fs.aware import AgentAwareFS

__all__ = [
    "AgentAwareFS",
    "connect_fs",
    "FileInfo",
    "FileMetadata",
    "FileSystem",
    "FSConfig",
    "get_current_fs",
    "IsolatedFS",
    "IsolatedFSConfig",
    "suspend",
    "patch",
    "VirtualFile",
    "VirtualFS",
    "VirtualFSConfig",
    "with_fs_context",
]
