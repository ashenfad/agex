"""
Hierarchical Agent Orchestration

An orchestrator agent delegates to specialist sub-agents for data generation
and visualization.

This example demonstrates how to build a multi-agent system by composing
agent tasks from different modules. It imports and uses the `make_data` task
from `data.py` and the `plot_data` task from `viz.py`, treating them as
callable functions for the orchestrator agent.

Bulk data flows between sub-agents without special handling.
"""

from data import make_data
from plotly.graph_objects import Figure
from viz import plot_data

from agex import Agent, connect_llm

orchestrator = Agent(
    name="orchestrator",
    primer="You orchestrate other agents to solve a problem.",
    llm_client=connect_llm(provider="openai", model="gpt-4.1-nano"),
)

# Give the orchestrator access to the specialist tasks.
# From the orchestrator's perspective, these are just functions it can call.
orchestrator.fn(make_data)
orchestrator.fn(plot_data)


@orchestrator.task
def idea_to_plot(idea: str) -> Figure:  # type: ignore[return-value]
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
