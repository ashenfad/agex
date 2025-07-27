"""
Plotly registration helpers for agex agents.

This module provides helper functions to register plotly classes
and methods with agents, including plotly express and graph objects.
"""

import warnings

from agex.agent import Agent

PLOTLY_EXCLUDE = [
    "_*",
    "*._*",
    "write_*",
    "to_html",
    "to_json",
    "write_html",
    "write_image",
    "show",
    "offline.*",
]

FIGURE_EXCLUDE = [
    "_*",
    "*._*",
    "write_*",
    "show",
    "print_grid",
]

TOOLS_EXCLUDE = [
    "_*",
    "*._*",
    "mpl_to_plotly",  # matplotlib conversion - probably not needed
    "get_config_*",  # server configuration
    "warning_*",  # internal warnings
]

IO_EXCLUDE = [
    "_*",
    "*._*",
    "write_*",
    "to_*",
    "show",
    "kaleido",  # image export engine
    "orca",  # legacy image export
    "base64_to_*",  # file conversion utilities
]


def register_plotly(agent: Agent) -> None:
    """Register plotly express, graph objects, tools, and templates with the agent."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Register plotly express for easy plotting
        agent.module(px, visibility="low", exclude=PLOTLY_EXCLUDE)

        # Register core graph objects for advanced plotting
        agent.module(go, visibility="low", exclude=PLOTLY_EXCLUDE)

        # Register the Figure class specifically for add_scatter, update_layout, etc.
        agent.cls(go.Figure, visibility="low", exclude=FIGURE_EXCLUDE)

        # Register tools for subplots and utilities
        import plotly.tools

        agent.module(plotly.tools, visibility="low", exclude=TOOLS_EXCLUDE)

        # Register IO module for templates and configuration
        import plotly.io

        agent.module(plotly.io, visibility="low", exclude=IO_EXCLUDE)

        # Register useful submodules
        import plotly.colors
        import plotly.figure_factory as ff

        agent.module(plotly.colors, visibility="low")
        agent.module(ff, visibility="low", exclude=PLOTLY_EXCLUDE)

    except ImportError:
        warnings.warn(
            "plotly not installed - skipping plotly registration", UserWarning
        )
        raise
