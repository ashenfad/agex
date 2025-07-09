"""
Numpy registration helpers for agex agents.

This module provides helper functions to register numpy classes
and methods with agents, including useful submodules.
"""

import warnings

from agex.agent import Agent

NUMPY_EXCLUDE = [
    "_*",
    "*._*",
    "load*",
    "save*",
    "fromfile",
    "tofile",
    "memmap",
    "DataSource*",
]


def register_numpy(agent: Agent) -> None:
    """Register numpy core classes and useful submodules with the agent."""
    try:
        import numpy as np

        # Register core numpy module
        agent.module(np, visibility="low", exclude=NUMPY_EXCLUDE)

        # Register useful submodules
        agent.module(
            np.random, visibility="low", exclude=["seed", "set_state", "get_state"]
        )
        agent.module(np.linalg, visibility="low")
        agent.module(np.fft, visibility="low")
        agent.module(np.ma, visibility="low")

    except ImportError:
        warnings.warn("numpy not installed - skipping numpy registration", UserWarning)
        raise
