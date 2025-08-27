"""
Numpy registration helpers for agex agents.

This module provides helper functions to register numpy classes
and methods with agents, including useful submodules.
"""

import warnings

from agex.agent import Agent

IO_EXCLUDE = [
    "load*",
    "save*",
    "fromfile",
    "tofile",
]

CORE_EXCLUDE = [
    "_*",
    "*._*",
    # Memory-mapped files
    "memmap",
    "DataSource*",
    # Unsafe random state manipulation from np.random
    "seed",
    "set_state",
    "get_state",
]


def register_numpy(agent: Agent, io_friendly: bool = False) -> None:
    """Register the entire numpy library recursively."""
    try:
        import numpy as np

        exclude = CORE_EXCLUDE
        if not io_friendly:
            exclude += IO_EXCLUDE
        agent.module(
            np,
            recursive=True,
            visibility="low",
            exclude=exclude,
        )

    except ImportError:
        warnings.warn("numpy not installed - skipping numpy registration", UserWarning)
        raise
