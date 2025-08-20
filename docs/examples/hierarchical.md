# Hierarchical Orchestration

Compose specialist agents into a workflow using the dual‑decorator pattern. In this example, an "orchestrator" agent will delegate tasks to a "data_maker" and a "plotty" agent to turn a high-level idea into a complete data visualization.

This pattern allows the orchestrator agent to treat specialist tasks like normal Python functions, passing real, complex objects like NumPy arrays and Plotly figures between them without any boilerplate.

## 1. Create Specialist Agents

First, we define our two specialist agents. The `data_maker` is an expert at creating datasets, and `plotty` is an expert at visualizing them. Each is given the specific libraries it needs to perform its role.

```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agex import Agent

# Define the data-making agent and give it numpy and random
data_maker = Agent(
    name="data_maker",
    primer="You excel at generating data via numpy.",
)
data_maker.module(np, recursive=True, visibility="low")

# Define the plotting agent and give it plotting modules
plotty = Agent(
    name="plotty",
    primer="You excel plotting data via plotly express.",
)
plotty.module(px, visibility="low")
plotty.module(go, visibility="low")
plotty.module(pd, visibility="low")
```

## 2. Define the Orchestrator

Next, we create the orchestrator agent. It doesn't need any specific libraries itself. Its job is to understand a high-level goal and delegate the sub-tasks to the correct specialists.

We use the dual-decorator pattern (`@orchestrator.fn` + `@specialist.task`) to make the specialist tasks available as simple functions that the orchestrator can call.

```python
# Define the orchestrator agent
orchestrator = Agent(
    name="orchestrator",
    primer="You orchestrate other agents to solve a problem.",
)

# Make the data_maker's task available to the orchestrator
@orchestrator.fn
@data_maker.task
def make_data(prompt: str) -> list[np.ndarray]:  # type: ignore[return-value]
    """Produce numpy arrays given the prompt."""
    pass

# Make the plotty's task available to the orchestrator
@orchestrator.fn
@plotty.task
def plot_data(prompt: str, data: list[np.ndarray]) -> go.Figure:  # type: ignore[return-value]
    """Produce a figure from numpy data given the prompt."""
    pass

# This is the main task that the orchestrator itself will implement
@orchestrator.task
def idea_to_plot(idea: str) -> go.Figure:  # type: ignore[return-value]
    """
    You are given an idea for a plot. You need to orchestrate the other agents to create the plot.
    """
    pass
```

## 3. Run the Orchestration

Now we can call the orchestrator's main task, `idea_to_plot`, with a high-level idea. The orchestrator will autonomously call `make_data` and then `plot_data`, passing the data between the specialists, to generate the final plot.

```python
idea = """
I'd like a plot that shows seasonal change over the years for umbrellas sold. The data should be artificial but realistic and span 10 years.
"""

plot = idea_to_plot(idea)
plot.show() # or plot.write_image(...)
```

## Why This is Compelling

-   **Natural Composition:** The dual-decorator pattern turns specialist agents into callable functions for the orchestrator. Function signatures are the contracts.
-   **Clear Separation of Concerns:** Each agent has a targeted primer and a specific set of capabilities, helping agents focus their skills.
-   **Seamless Data Flow:** Complex, rich objects like NumPy arrays and Plotly Figures flow directly between agents with no need for intermediate JSON serialization or wrapper tools.

---

Source: [https://github.com/ashenfad/agex/blob/main/examples/hierarchical.py](https://github.com/ashenfad/agex/blob/main/examples/hierarchical.py)
