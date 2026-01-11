"""FileSystem support for agex agents.

This module provides both virtual (in-memory) and isolated (restricted real)
filesystem access for agents via standard Python file operations.

Public API:
    connect_fs: Factory function for configuring filesystem access
    VirtualFS: State-backed virtual filesystem implementation
    IsolatedFS: Real filesystem with path restriction
    with_virtual_fs: Context manager for virtual FS
    with_isolated_fs: Context manager for isolated FS
    swap_agent_fs_functions: Swap registered fs functions with FS-aware versions
"""

from agex.fs.aware import AgentAwareFS
from agex.fs.config import (
    FSConfig,
    IsolatedFSConfig,
    VirtualFSConfig,
    connect_fs,
)
from agex.fs.isolated import IsolatedFS
from agex.fs.patching import (
    swap_agent_fs_functions,
    with_fs_context,
    with_isolated_fs,
    with_virtual_fs,
)
from agex.fs.virtual import FileInfo, FileMetadata, VirtualFile, VirtualFS

__all__ = [
    "AgentAwareFS",
    "connect_fs",
    "FSConfig",
    "FileInfo",
    "FileMetadata",
    "IsolatedFS",
    "IsolatedFSConfig",
    "swap_agent_fs_functions",
    "VirtualFile",
    "VirtualFS",
    "VirtualFSConfig",
    "with_fs_context",
    "with_isolated_fs",
    "with_virtual_fs",
]
