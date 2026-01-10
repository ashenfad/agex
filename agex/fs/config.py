"""Configuration for agent filesystem access.

Provides configuration dataclasses and connect_fs factory function for
configuring agent filesystem access (virtual or isolated).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class VirtualFSConfig:
    """Configuration for virtual (in-memory) filesystem.

    Attributes:
        type: Always "virtual".
    """

    type: Literal["virtual"] = "virtual"


@dataclass
class IsolatedFSConfig:
    """Configuration for isolated (real) filesystem with path restriction.

    Attributes:
        type: Always "isolated".
        root: Absolute path to root directory (all file operations restricted to this path).
        tracking: Whether to track file changes and emit FileEvents (default: False).
        per_session: Whether to create session subdirectories for isolated filesystems (default: False).
    """

    type: Literal["isolated"] = "isolated"
    root: str = ""
    tracking: bool = False
    per_session: bool = False


# Type alias for all filesystem configs
FSConfig = VirtualFSConfig | IsolatedFSConfig


def connect_fs(
    type: Literal["virtual", "isolated"] = "virtual",
    **kwargs,
) -> FSConfig:
    """Configure filesystem access for agents.

    Creates a filesystem configuration that can be passed to Agent().

    Args:
        type: Filesystem type.
            - "virtual": In-memory filesystem backed by agent state.
                        Files persist with state and participate in versioning.
            - "isolated": Real filesystem restricted to a directory.
                         Requires 'root' argument.
        **kwargs: Additional configuration for the filesystem type.
            For type="isolated":
                - root (str): Required. Absolute path to root directory.
                - tracking (bool): Optional. Track file changes (default: False).

    Returns:
        FSConfig for Agent initialization.

    Examples:
        Virtual filesystem:
        >>> from agex import Agent, connect_fs, connect_state
        >>> agent = Agent(
        ...     state=connect_state(type="versioned", storage="disk", path="/tmp/state"),
        ...     fs=connect_fs(type="virtual"),
        ... )

        Isolated filesystem:
        >>> agent = Agent(
        ...     fs=connect_fs(type="isolated", root="/path/to/project", tracking=True),
        ... )
    """
    if type == "virtual":
        if kwargs:
            raise ValueError(
                f"Unexpected arguments for virtual fs: {list(kwargs.keys())}"
            )
        return VirtualFSConfig(type=type)

    elif type == "isolated":
        # Extract isolated-specific kwargs
        root = kwargs.pop("root", "")
        tracking = kwargs.pop("tracking", False)
        per_session = kwargs.pop("per_session", False)

        if kwargs:
            raise ValueError(
                f"Unexpected arguments for isolated fs: {list(kwargs.keys())}"
            )

        if not root:
            raise ValueError("Isolated filesystem requires 'root' parameter")

        return IsolatedFSConfig(root=root, tracking=tracking, per_session=per_session)

    else:
        raise ValueError(
            f"Unsupported filesystem type: {type}. Use 'virtual' or 'isolated'."
        )
