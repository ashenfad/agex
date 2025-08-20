"""
Hierarchical Agent Orchestration

An orchestrator agent delegates to specialist sub-agents for data generation
and visualization. A sub-agents task is the orchestrator's fn. The signature
is the contract between them.

Bulk data and plots flow between agents without special handling.
"""

import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from agex import Agent, connect_llm

llm_client = connect_llm(provider="openai", model="gpt-4.1-nano")


# define the data-making agent and give it numpy and random
data_maker = Agent(
    name="data_maker",
    primer="You excel at generating data via numpy.",
    llm_client=llm_client,
)

data_maker.module(np, recursive=True, visibility="low")
data_maker.module(random, visibility="low")


# define the plotting agent and give it a few modules
plotty = Agent(
    name="plotty",
    primer="You excel plotting data via plotly express.",
    llm_client=llm_client,
)

plotty.module(np, recursive=True, visibility="low")
plotty.module(px, visibility="low")
plotty.module(go, visibility="low")
plotty.module(pd, visibility="low")

# define the orchestrator agent, no special modules are needed
orchestrator = Agent(
    name="orchestrator",
    primer="You orchestrate other agents to solve a problem.",
    llm_client=llm_client,
)


# Define task fns & give the orchestrator access to the specialist tasks


@orchestrator.fn
@data_maker.task
def make_data(prompt: str) -> list[np.ndarray]:  # type: ignore[return-value]
    """Produce numpy arrays given the prompt."""
    pass


@orchestrator.fn
@plotty.task
def plot_data(prompt: str, data: list[np.ndarray]) -> go.Figure:  # type: ignore[return-value]
    """Produce a figure from numpy data given the prompt."""
    pass


@orchestrator.task
def idea_to_plot(idea: str) -> go.Figure:  # type: ignore[return-value]
    """
    You are given an idea for a plot. You need to orchestrate the other agents to create the plot.
    """
    pass


def main():
    # ask the orchestrator to create a plot from an idea, it will delegate to sub-agents
    idea = """
    I'd like a plot that shows seasonal change over the years for umbrellas sold. The data
    should be artificial but realistic and span 10 years.
    """

    plot = idea_to_plot(idea)
    plot.write_image("examples/seasonal.png")
    # see examples/seasonal.png


if __name__ == "__main__":
    # Run with: python examples/hierarchical.py OR python -m examples.hierarchical
    main()
