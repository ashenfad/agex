import pandas as pd
from plotly.graph_objects import Figure

from agex import Agent

# --- The Database Expert ---
db_expert = Agent(name="db_expert")
db_expert.module(pd)  # It also knows pandas


@db_expert.task
def get_sales_data(query: str, db_connection) -> pd.DataFrame:
    """Runs a SQL query and returns the result as a pandas DataFrame."""
    pass


# --- The Visualization Expert ---
viz_expert = Agent(name="viz_expert")
# ... register plotly ...


@viz_expert.task
def plot_dataframe(df: pd.DataFrame, title: str) -> Figure:
    """Takes a DataFrame and returns a plot."""
    pass
