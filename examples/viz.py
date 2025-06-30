import numpy as np
import pandas
import plotly.express
from plotly.graph_objects import Figure

from agex import Agent

viz = Agent(
    name="viz",
    primer="You excel plotting data via plotly express.",
    timeout_seconds=20,
)

viz.module(np, visibility="low")
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

# reuse our data agent to get signals
# data = make_data(gen_data_request)

# plot = plot_data(viz_request, data)
# plot.show()
