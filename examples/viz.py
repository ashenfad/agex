"""
Data Visualization

Agent works with complex NumPy arrays and generates Plotly visualizations.
Bulk data flows between agents without special handling.

Note: This example was tested with gpt-4.1-nano to demonstrate that
even smaller LLMs can effectively use the agex framework.
"""

import numpy as np
import pandas
import plotly.express
from plotly.graph_objects import Figure

from agex import Agent
from examples.data import gen_data_request, make_data

viz = Agent(
    name="viz",
    primer="You excel plotting data via plotly express.",
)

viz.module(np, visibility="low")
viz.module(np.random, visibility="low")
viz.module(plotly.express, visibility="low")
viz.module(plotly.graph_objs, visibility="low")
viz.module(pandas, visibility="low")


@viz.task
def plot_data(prompt: str, data: list[np.ndarray]) -> Figure:  # type: ignore[return-value]
    """Produce a figure from numpy data given the prompt."""
    pass


viz_request = """
Each numpy array represents a signal from a manufacturing process. Please plot all the signals
as a scatter plot, the x-axis is time and every point in the array corresponds to a second.
The y-axis is the signal.
"""


def example():
    """
    We gather bulk data from an agent and hand it to another for plotting.
    """

    # reuse our data agent to get signals
    data = make_data(gen_data_request)

    # share that bulk data with another agent to create a plot
    plot = plot_data(viz_request, data)

    # there's a cool plot with some abberations... really!
    plot.show()
    # see examples/process.png
