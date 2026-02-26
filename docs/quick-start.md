# Quick Start

This guide walks you through the core concepts of agex with hands-on examples.

## Your First Agent

Install agex with your preferred provider (`pip install "agex[openai]"`, `"agex[anthropic]"`, or `"agex[gemini]"`) and create an agent:

```python
import math
from agex import Agent, connect_llm

# Create an LLM client (or use AGEX_LLM_PROVIDER/AGEX_LLM_MODEL env vars)
llm = connect_llm(provider="openai", model="gpt-4.1-nano")

agent = Agent(
    primer="You are great at solving math problems.",
    llm=llm
)

# Give it access to math functions
agent.module(math, visibility="medium")

# Define a task (empty body - agent implements it)
@agent.task
def solve_equation(equation: str) -> float:  # type: ignore[return-value]
    """Solve a mathematical equation and return the result."""
    pass

# Use it
result = solve_equation("What is the square root of 256, multiplied by pi?")
print(result)  # 50.26548245743669
```

> **Note**: `# type: ignore[return-value]` silences type checkers — the agent provides the implementation at runtime.

## Custom Functions

You can register your own functions as agent capabilities:

```python
from agex import Agent

agent = Agent()

@agent.fn
def calculate_compound_interest(principal: float, rate: float, years: int) -> float:
    """Calculate compound interest."""
    return principal * ((1 + rate) ** years)

@agent.task
def investment_analysis(amount: float, annual_rate: float, years: int) -> str:  # type: ignore[return-value]
    """Analyze an investment scenario with explanations."""
    pass

analysis = investment_analysis(10000, 0.07, 10)
print(analysis)
```

The agent can now use your custom `calculate_compound_interest` function while generating its response.

## Working with Complex Data

Agents can work with rich Python objects like numpy arrays and pandas DataFrames:

```python
import numpy as np
from agex import Agent

data_agent = Agent(primer="You excel at generating data via numpy.")
data_agent.module(np, visibility="low")
data_agent.module(np.random, visibility="low")

@data_agent.task
def create_dataset(description: str) -> list[np.ndarray]:  # type: ignore[return-value]
    """Generate numpy arrays based on the description."""
    pass

# Agent returns real numpy arrays you can use immediately
signals = create_dataset("Generate 5 sine waves with different frequencies")
print(f"Created {len(signals)} arrays, first one shape: {signals[0].shape}")

# Use the data with regular Python code
combined = np.concatenate(signals)
```

## Persistent State

For agents that need to remember across calls, use `connect_state`:

```python
from agex import Agent, connect_state

comedian = Agent(
    primer="You're a comedian who builds elaborate jokes over time.",
    state=connect_state(type="versioned", storage="memory"),
)

@comedian.task
def workshop_joke(prompt: str) -> str:  # type: ignore[return-value]
    """Build on the ongoing joke based on the prompt."""
    pass

# Agent builds an elaborate joke across multiple calls (state is managed internally)
setup = workshop_joke("Start a joke about a programmer and a fish")
buildup = workshop_joke("Add more detail about their meeting")
punchline = workshop_joke("Deliver the punchline!")

print(f"{setup}\n{buildup}\n{punchline}")
```

## Hierarchical Multi-Agent Orchestration

Create specialized agents that work together using the dual-decorator pattern:

```python
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure
from agex import Agent

# Create specialized agents
data_generator = Agent(name="data_generator", primer="You excel at generating data.")
visualizer = Agent(name="visualizer", primer="You excel at creating plots.")
orchestrator = Agent(name="orchestrator", primer="You coordinate other agents.")

# Give agents their required capabilities
data_generator.module(np, visibility="low")
visualizer.module(px, visibility="low")

# Dual-decorator pattern: orchestrator can call specialist tasks
@orchestrator.fn
@data_generator.task
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

# The orchestrator delegates to specialists automatically
plot = idea_to_visualization("Show seasonal trends in sales data over 3 years")
plot.show()
```

**Key concept:**

- **Dual decorators**: `@orchestrator.fn` + `@specialist.task` creates hierarchical agent flows where orchestrator agents can call specialist agents as functions

## Peer Collaboration

Agents can also collaborate as peers using standard Python control flow:

```python
# Iterative improvement between agents
content = create_content("python decorators")
while (review := review_content(content)).quality != "good":
    content = improve_content(content, review.feedback)
```

For a complete example with dataclass-based reviews and structured feedback, see the **[Examples](./examples/overview.md)**.

## Event Monitoring

Every task supports an `on_event` callback for real-time visibility into agent reasoning:

```python
from agex import pprint_events

# Stream events with built-in formatting
result = analyze_data([1, 5, 3], on_event=pprint_events)
```

Or write a custom handler to capture specific event types:

```python
from agex import ActionEvent

def capture_actions(event):
    if isinstance(event, ActionEvent):
        print(f"Thinking: {event.thinking}")
        print(f"Code: {event.code}")

result = analyze_data([1, 5, 3], on_event=capture_actions)
```

See the **[Events API](./api/events.md)** for all event types and patterns.


## Async Tasks

Define the task as `async def` and `await` the result:

```python
@agent.task
async def analyze_data(data: list[int]) -> dict:  # type: ignore[return-value]
    """Analyze a list of numbers."""
    pass

result = await analyze_data([1, 2, 3, 4, 5])
```

All features (`on_event`, `on_token`, state) work with async tasks.

## Task Errors

Agents may refuse tasks by raising a TaskFail or a TaskClarify. A task
may also fail if it exceeds an iteration limit:

```python
from agex import Agent, TaskFail, TaskClarify, TaskTimeout

agent = Agent()

@agent.task
def risky_task(input_data: str) -> str:  # type: ignore[return-value]
    """A task that might fail or need clarification."""
    pass

try:
    result = risky_task("ambiguous input")
    print(f"Success: {result}")
except TaskFail as e:
    print(f"Task failed: {e.message}")
except TaskClarify as e:
    print(f"Task needs clarification: {e.message}")
except TaskTimeout as e:
    print(f"Task exceeded max iterations: {e.message}")
```

## Next Steps

- **[API Reference](./api/overview.md)** - Complete documentation for all agex APIs
- **[Events API](./api/events.md)** - Comprehensive guide to event monitoring and debugging
- **[Examples](./examples/overview.md)** - Real-world examples showing advanced patterns
- **[The Big Picture](./concepts/big-picture.md)** - Framework philosophy and design principles
