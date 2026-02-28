"""FileSystem support for agex agents.

This module provides both virtual (in-memory) and isolated (restricted real)
filesystem access for agents via standard Python file operations.

Core FS functionality is provided by the monkeyfs library. This module
re-exports its public API and adds agex-specific configuration:
    AgentAwareFS: FS wrapper that emits events for agent visibility
    connect_fs: Factory for filesystem configuration
"""

from dataclasses import dataclass, field
from typing import Literal

# Re-export monkeyfs public API
from monkeyfs import (
    FileInfo,
    FileMetadata,
    FileSystem,
    IsolatedFS,
    VirtualFS,
    patch,
    suspend,
)

# agex-specific
from agex.fs.aware import AgentAwareFS


@dataclass
class VirtualFSConfig:
    """Configuration for virtual (in-memory) filesystem.

    Attributes:
        type: Always "virtual".
        max_size_mb: Maximum total size of all files in megabytes.
            None means unlimited.
    """

    type: Literal["virtual"] = "virtual"
    max_size_mb: int | None = None


@dataclass
class IsolatedFSConfig:
    """Configuration for isolated (real) filesystem with path restriction.

    Attributes:
        root: Absolute path to root directory.
        type: Always "isolated".
        per_session: Create session subdirectories (default: False).
    """

    root: str
    type: Literal["isolated"] = "isolated"
    per_session: bool = field(default=False)


# Type alias for all filesystem configs
FSConfig = VirtualFSConfig | IsolatedFSConfig


def connect_fs(
    type: Literal["virtual", "isolated"] = "virtual",
    **kwargs,
) -> FSConfig:
    """Configure filesystem access.

    Args:
        type: FileSystem type.
            - "virtual": In-memory filesystem backed by a mapping.
            - "isolated": Real filesystem restricted to a directory.
        **kwargs: Additional configuration for the filesystem type.
            For type="virtual":
                - max_size_mb (int): Optional. Max total file size in MB.
            For type="isolated":
                - root (str): Required. Absolute path to root directory.
                - per_session (bool): Create session subdirectories (default: False).

    Returns:
        FSConfig for deferred filesystem instantiation.
    """
    if type == "virtual":
        max_size_mb = kwargs.pop("max_size_mb", None)
        if kwargs:
            raise ValueError(
                f"Unexpected arguments for virtual fs: {list(kwargs.keys())}"
            )
        return VirtualFSConfig(type=type, max_size_mb=max_size_mb)

    elif type == "isolated":
        root = kwargs.pop("root", "")
        per_session = kwargs.pop("per_session", False)

        if kwargs:
            raise ValueError(
                f"Unexpected arguments for isolated fs: {list(kwargs.keys())}"
            )

        if not root:
            raise ValueError("Isolated filesystem requires 'root' parameter")

        return IsolatedFSConfig(root=root, per_session=per_session)

    else:
        raise ValueError(
            f"Unsupported filesystem type: {type}. Use 'virtual' or 'isolated'."
        )


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
    "VirtualFS",
    "VirtualFSConfig",
]
