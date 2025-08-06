"""
Plotly registration helpers for agex agents.

This module provides helper functions to register plotly classes
and methods with agents, including plotly express and graph objects.
"""

import warnings

from agex.agent import Agent

PLOTLY_EXCLUDE = [
    # General exclusions
    "_*",
    "*._*",
    # File I/O and serialization
    "write_*",
    "to_*",
    "to_html",
    "to_json",
    "write_html",
    "write_image",
    "base64_to_*",
    # Display and export
    "show",
    "plot",  # Blocks plotly.offline.plot
    "iplot",  # Blocks plotly.offline.iplot
    "kaleido",
    "orca",
    "print_grid",
    # Unneeded tools and configuration
    "mpl_to_plotly",
    "get_config_*",
    "warning_*",
]


def register_plotly(agent: Agent) -> None:
    """Register the entire plotly library recursively."""
    try:
        import plotly

        agent.module(
            plotly,
            recursive=True,
            visibility="low",
            exclude=PLOTLY_EXCLUDE,
        )

    except ImportError:
        warnings.warn(
            "plotly not installed - skipping plotly registration", UserWarning
        )
        raise
