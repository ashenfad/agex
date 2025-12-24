"""
Async Hierarchical Agent Orchestration

Demonstrates async multi-agent orchestration where an orchestrator agent
delegates to specialist sub-agents for data generation and visualization.

Agent-generated code remains synchronous while the framework handles
async transparently.

Usage: python examples/hierarchical_async.py
"""

import asyncio
import random

import numpy as np
import plotly.graph_objects as go

from agex import Agent, connect_llm, pprint_tokens
from agex.helpers import register_numpy, register_pandas, register_plotly

llm = connect_llm(provider="gemini", model="gemini-3-flash-preview")


# define the data-making agent and give it numpy and random
data_maker = Agent(
    name="data_maker",
    primer="You excel at generating data via numpy.",
    llm=llm,
)

data_maker.module(np, recursive=True, visibility="low")
data_maker.module(random, visibility="low")


# define the plotting agent and give it a few modules
plotty = Agent(
    name="plotty",
    primer="You excel plotting data via plotly express.",
    llm=llm,
)

# use helpers for our plotting agent
register_plotly(plotty)
register_numpy(plotty)
register_pandas(plotty)

# define the orchestrator agent, no special modules are needed
orchestrator = Agent(
    name="orchestrator",
    primer="You orchestrate other agents to solve a problem. Don't use numpy or plotly directly. Call 'make_data' and 'plot_data' to spawn sub-agent work.",
    llm=llm,
)


# Define async task fns & give the orchestrator access to the specialist tasks


@orchestrator.fn
@data_maker.task
async def make_data(prompt: str) -> list[np.ndarray]:  # type: ignore[return-value]
    """Produce numpy arrays given the prompt."""
    pass


@orchestrator.fn
@plotty.task
async def plot_data(prompt: str, data: list[np.ndarray]) -> go.Figure:  # type: ignore[return-value]
    """Produce a figure from numpy data given the prompt."""
    pass


@orchestrator.task
async def idea_to_plot(idea: str) -> go.Figure:  # type: ignore[return-value]
    """
    You are given an idea for a plot. You need to orchestrate the other agents to create the plot.
    """
    pass


async def main():
    # ask the orchestrator to create a plot from an idea, it will delegate to sub-agents
    idea = """
    I'd like a plot that shows seasonal change over the years for umbrellas sold. The data
    should be artificial but realistic and span 10 years.
    """

    print("\nPROMPT:", idea)
    plot = await idea_to_plot(idea, on_token=pprint_tokens)
    plot.write_image("examples/seasonal_async.png")
    # see examples/seasonal_async.png
    print("Plot saved to examples/seasonal_async.png")


if __name__ == "__main__":
    # Run with: python examples/hierarchical_async.py
    asyncio.run(main())
