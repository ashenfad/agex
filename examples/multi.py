from plotly.graph_objects import Figure

from agex import Agent
from examples.data import make_data
from examples.viz import plot_data

orchestrator = Agent(
    name="orchestrator",
    primer="You orchestrate other agents to solve a problem.",
    timeout_seconds=20,
)

# add sub-agents as functions (aka tools)
orchestrator.fn(make_data)
orchestrator.fn(plot_data)


@orchestrator.task
def idea_to_plot(idea: str) -> Figure:  # type: ignore[return-value]
    """
    You are given an idea for a plot. You need to orchestrate the other agents to create the plot.
    """
    pass


def example():
    """
    Ask the orchestrator to create a plot from an idea. It will delegate to sub-agents
    to create artifical data and then plot it.
    """

    idea = """
    I'd like a plot that shows seasonal change over the years for umbrellas sold. The data
    should be artificial but realistic and span 10 years.
    """

    plot = idea_to_plot(idea)
    plot.show()
