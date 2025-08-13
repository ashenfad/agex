"""
Pandas registration helpers for agex agents.

This module provides helper functions to register pandas classes
and methods with agents, including internal accessor classes.
"""

import warnings

from agex.agent import Agent

PANDAS_EXCLUDE = [
    "_*",
    "*._*",
    "read_*",
    "DataFrame.eval",
    "DataFrame.to_*",
    "pandas.io*",
    "pandas.core*",
    "pandas.plotting*",
    "pandas.testing*",
    "pandas.util*",
]


def register_pandas(agent: Agent) -> None:
    """Register pandas and its submodules recursively."""
    try:
        import pandas as pd

        agent.module(
            pd,
            recursive=True,
            visibility="low",
            exclude=PANDAS_EXCLUDE,
        )

    except ImportError:
        warnings.warn(
            "pandas not installed - skipping pandas registration", UserWarning
        )
        raise
