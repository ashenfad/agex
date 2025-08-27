"""
Pandas registration helpers for agex agents.

This module provides helper functions to register pandas classes
and methods with agents, including internal accessor classes.
"""

import warnings

from agex.agent import Agent

IO_EXCLUDE = [
    "read_*",
    "pandas.io*",
    "DataFrame.to_*",
]

CORE_EXCLUDE = [
    "_*",
    "*._*",
    "DataFrame.eval",
    "pandas.core*",
    "pandas.plotting*",
    "pandas.testing*",
    "pandas.util*",
]


def register_pandas(agent: Agent, io_friendly: bool = False) -> None:
    """Register pandas and its submodules recursively."""
    try:
        import pandas as pd

        exclude = CORE_EXCLUDE
        if not io_friendly:
            exclude += IO_EXCLUDE
        agent.module(pd, recursive=True, visibility="low", exclude=exclude)

    except ImportError:
        warnings.warn(
            "pandas not installed - skipping pandas registration", UserWarning
        )
        raise
