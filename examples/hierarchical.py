"""
Hierarchical Agent Orchestration (Dual-Decorator Pattern)

This example demonstrates the recommended pattern for building hierarchical multi-agent
systems using the dual-decorator `@agent.fn` and `@agent.task`.

An orchestrator agent is given access to two specialist agents: one for generating
data and another for creating visualizations. The orchestrator's task is to take a
high-level idea and use the specialists to produce a final plot.

This pattern is useful for breaking down complex problems into smaller, manageable
tasks that can be handled by specialized agents, all coordinated by a higher-level
orchestrator.
"""

import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure

from agex import Agent

# 1. Create the specialist agents
data_generator = Agent(
    name="data_generator",
    primer="You are an expert at generating synthetic datasets using NumPy.",
)
visualizer = Agent(
    name="visualizer",
    primer="You are an expert at creating beautiful visualizations using Plotly.",
)
orchestrator = Agent(
    name="orchestrator",
    primer="You solve problems by delegating tasks to specialist agents.",
)

# 2. Give specialists their required tools
data_generator.module(np)
visualizer.module(px)
visualizer.module(np)


# 3. Define the specialist tasks and expose them to the orchestrator
#    using the dual-decorator pattern.
@orchestrator.fn(docstring="Generates synthetic data based on a description.")
@data_generator.task
def make_data(description: str) -> np.ndarray:  # type: ignore[return-value]
    """Generate a NumPy array containing synthetic data that matches the description."""
    pass


@orchestrator.fn(docstring="Creates a plot from a NumPy array.")
@visualizer.task
def create_plot(data: np.ndarray, title: str) -> Figure:  # type: ignore[return-value]
    """Create a Plotly figure from the provided NumPy data with a given title."""
    pass


# 4. Define the orchestrator's high-level task. The orchestrator will
#    automatically know how to call `make_data` and `create_plot`.
@orchestrator.task
def idea_to_visualization(idea: str) -> Figure:  # type: ignore[return-value]
    """
    Take a high-level idea from the user and use the available specialist
    functions to generate the data and create a final visualization.
    """
    pass


def main():
    """
    Execute the orchestration task.
    """
    idea = "Create a scatter plot of 100 random 2D points."
    plot = idea_to_visualization(idea)

    # Verify that we received a real Plotly figure
    print(f"Successfully created a {type(plot).__name__} object.")
    print(f"Figure contains {len(plot.data)} trace(s).")  # type: ignore
    # You can optionally show the plot or save it to a file
    # plot.show()
    # plot.write_image("examples/hierarchical_plot.png")


if __name__ == "__main__":
    main()
