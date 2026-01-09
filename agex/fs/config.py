"""Configuration for virtual filesystem.

Provides the FSConfig dataclass and connect_fs factory function for
configuring agent filesystem access.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FSConfig:
    """Configuration for agent filesystem access.

    Attributes:
        type: Filesystem type. Currently only "virtual" is supported.
    """

    type: Literal["virtual"] = "virtual"


def connect_fs(
    type: Literal["virtual"] = "virtual",
) -> FSConfig:
    """Configure filesystem access for agents.

    Creates a filesystem configuration that can be passed to Agent().

    Args:
        type: Filesystem type.
            - "virtual": In-memory filesystem backed by agent state.
                        Files persist with state and participate in versioning.

    Returns:
        FSConfig for Agent initialization.

    Example:
        >>> from agex import Agent, connect_fs, connect_state
        >>> agent = Agent(
        ...     state=connect_state(type="versioned", storage="disk", path="/tmp/state"),
        ...     fs=connect_fs(type="virtual"),
        ... )
        >>> fs = agent.fs()
        >>> fs.write("data.csv", b"a,b,c\\n1,2,3")
    """
    if type != "virtual":
        raise ValueError(f"Unsupported filesystem type: {type}. Use 'virtual'.")

    return FSConfig(type=type)
