# Hierarchical Orchestration

Compose specialists into an orchestrator using the dual‑decorator pattern. Treat specialist tasks like normal Python functions and pass real objects between agents.

## Specialists

```python
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure
from agex import Agent

# Create specialized agents
data_generator = Agent(name="data_generator", primer="You excel at generating data.")
visualizer = Agent(name="visualizer", primer="You excel at creating plots.")

# Give them their capabilities
data_generator.module(np, visibility="low")
visualizer.module(px, visibility="low")
```

## Orchestrator and dual‑decorated tasks

```python
orchestrator = Agent(name="orchestrator", primer="You coordinate other agents.")

@orchestrator.fn  # orchestrator can call this like a function
@data_generator.task  # implemented by the data_generator agent
def generate_data(description: str) -> list[np.ndarray]:  # type: ignore[return-value]
    """Generate synthetic datasets matching the description."""
    pass

@orchestrator.fn
@visualizer.task
def create_plot(data: list[np.ndarray]) -> Figure:  # type: ignore[return-value]
    """Turn numpy arrays into an interactive plot."""
    pass

@orchestrator.task
def idea_to_visualization(idea: str) -> Figure:  # type: ignore[return-value]
    """Turn a visualization idea into a complete data plot."""
    pass
```

## Use it

```python
idea = "Show seasonal trends in sales data over 3 years"
plot = idea_to_visualization(idea)
plot.show()  # or write_image(...)
```

## Why this is compelling
- Natural composition: specialists become callable functions for the orchestrator.
- Bulk object passing: large NumPy arrays flow between agents without JSON serialization.
- Clear separation of concerns with tiny primers per role.

—

Sources:

- Orchestration: [https://github.com/ashenfad/agex/blob/main/examples/hierarchical.py](https://github.com/ashenfad/agex/blob/main/examples/hierarchical.py)
- Data: [https://github.com/ashenfad/agex/blob/main/examples/data.py](https://github.com/ashenfad/agex/blob/main/examples/data.py)
- Viz: [https://github.com/ashenfad/agex/blob/main/examples/viz.py](https://github.com/ashenfad/agex/blob/main/examples/viz.py)
