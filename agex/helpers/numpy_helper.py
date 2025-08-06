"""
Numpy registration helpers for agex agents.

This module provides helper functions to register numpy classes
and methods with agents, including useful submodules.
"""

import warnings

from agex.agent import Agent

NUMPY_EXCLUDE = [
    # General private/internal members
    "_*",
    "*._*",
    # File I/O
    "load*",
    "save*",
    "fromfile",
    "tofile",
    # Memory-mapped files
    "memmap",
    "DataSource*",
    # Unsafe random state manipulation from np.random
    "seed",
    "set_state",
    "get_state",
]


def register_numpy(agent: Agent) -> None:
    """Register the entire numpy library recursively."""
    try:
        import numpy as np

        agent.module(
            np,
            recursive=True,
            visibility="low",
            exclude=NUMPY_EXCLUDE,
        )

    except ImportError:
        warnings.warn("numpy not installed - skipping numpy registration", UserWarning)
        raise
