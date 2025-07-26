"""
Pandas registration helpers for agex agents.

This module provides helper functions to register pandas classes
and methods with agents, including internal accessor classes.
"""

import warnings

from pandas.core.window.rolling import Rolling

from agex.agent import Agent

PANDAS_EXCLUDE = [
    "_*",
    "*._*",
    "read_*",
    "DataFrame.eval",
    "DataFrame.to_pickle",
    "DataFrame.to_csv",
    "DataFrame.to_clipboard",
    "DataFrame.to_excel",
    "DataFrame.to_json",
    "DataFrame.to_html",
    "DataFrame.to_xml",
    "DataFrame.to_latex",
    "DataFrame.to_feather",
    "DataFrame.to_parquet",
    "DataFrame.to_sql",
    "DataFrame.to_stata",
]


def register_pandas(agent: Agent) -> None:
    """Register pandas core classes and accessor classes with the agent."""
    try:
        import pandas as pd

        # Register core pandas classes
        agent.module(pd, visibility="low", exclude=PANDAS_EXCLUDE)
        agent.module(pd.api.types, visibility="low")

        pd.Series
        # Register accessor classes for .dt, .str, .cat
        from pandas.core.arrays.categorical import CategoricalAccessor
        from pandas.core.indexes.accessors import (
            DatetimeProperties,
            PeriodProperties,
            TimedeltaProperties,
        )
        from pandas.core.strings.accessor import StringMethods

        agent.cls(DatetimeProperties, visibility="low")
        agent.cls(TimedeltaProperties, visibility="low")
        agent.cls(PeriodProperties, visibility="low")
        agent.cls(StringMethods, visibility="low")
        agent.cls(CategoricalAccessor, visibility="low")
        agent.cls(Rolling, visibility="low")

    except ImportError:
        warnings.warn(
            "pandas not installed - skipping pandas registration", UserWarning
        )
        raise
