"""
Hierarchical Agent Orchestration

An orchestrator agent delegates to specialist sub-agents for data generation
and visualization. Bulk data flows between sub-agents without special handling.
"""

from plotly.graph_objects import Figure

from agex import Agent
from examples.data import make_data
from examples.viz import plot_data

orchestrator = Agent(
    name="orchestrator",
    primer="You orchestrate other agents to solve a problem.",
)

# add sub-agents as functions for orchestrator
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
    plot.show()
    # see examples/seasonal.png


if __name__ == "__main__":
    # Run with: python examples/multi.py OR python -m examples.multi
    main()
