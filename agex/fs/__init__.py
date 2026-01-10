"""Virtual filesystem support for agex agents.

This module provides a state-backed virtual filesystem that agents can access
via standard Python file operations (open, listdir, etc.).

Public API:
    connect_fs: Factory function for configuring filesystem access
    VirtualFS: State-backed virtual filesystem implementation
    with_virtual_fs: Context manager for setting VFS in current async context
    swap_agent_fs_functions: Swap registered fs functions with VFS-aware versions
"""

from agex.fs.aware import AgentAwareVFS
from agex.fs.config import FSConfig, connect_fs
from agex.fs.patching import swap_agent_fs_functions, with_virtual_fs
from agex.fs.virtual import FileInfo, FileMetadata, VirtualFile, VirtualFS

__all__ = [
    "AgentAwareVFS",
    "connect_fs",
    "FSConfig",
    "FileInfo",
    "FileMetadata",
    "VirtualFS",
    "VirtualFile",
    "with_virtual_fs",
    "swap_agent_fs_functions",
]
