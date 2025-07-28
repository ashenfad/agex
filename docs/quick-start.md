# Quick Start Guide

This guide walks you through the core concepts of agex with hands-on examples. By the end, you'll understand how to create agents, register capabilities, and build multi-agent workflows.

## Basic Setup

First, install agex with your preferred LLM provider:

```bash
# Install with specific provider
pip install "agex[openai]"      # For OpenAI models
pip install "agex[anthropic]"   # For Anthropic Claude models  
pip install "agex[gemini]"      # For Google Gemini models

# Or install with all providers
pip install "agex[all-providers]"

# Or install just the core (dummy provider only)
pip install agex
```

Configure your LLM either via environment variables:

```bash
export AGEX_LLM_PROVIDER=openai
export AGEX_LLM_MODEL=gpt-4
```

Or programmatically:

```python
from agex import configure_llm

# Configure your LLM (OpenAI, Anthropic, or Gemini)
configure_llm(provider="openai", model="gpt-4")
configure_llm(provider="anthropic", model="claude-3-sonnet-20240229")
configure_llm(provider="gemini", model="gemini-1.5-flash")
```

## 1. Your First Agent

Let's start with a simple agent that can do math:

```python
import math
from agex import Agent

# Create an agent
agent = Agent(primer="You are great at solving math problems.")

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

**Key concepts:**

- **`Agent(primer=...)`**: Creates an agent with behavioral instructions
- **`agent.module()`**: Exposes existing Python modules to the agent
- **`@agent.task`**: Defines what you want accomplished (agent provides implementation)

## 2. Custom Functions

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

## 3. Working with Complex Data

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

## 4. Persistent State

For agents that need to remember across calls, use `Versioned` state:

```python
from agex import Agent, Versioned

comedian = Agent(primer="You're a comedian who builds elaborate jokes over time.")

@comedian.task
def workshop_joke(prompt: str, state: Versioned) -> str:  # type: ignore[return-value]
    """Build on the ongoing joke based on the prompt."""
    pass

# Agent builds an elaborate joke across multiple calls
state = Versioned()
setup = workshop_joke("Start a joke about a programmer and a fish", state=state)
buildup = workshop_joke("Add more detail about their meeting", state=state)  
punchline = workshop_joke("Deliver the punchline!", state=state)

print(f"{setup}\n{buildup}\n{punchline}")
```

## 5. Hierarchical Multi-Agent Orchestration

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

## 6. Other Agent Collaboration Patterns

Beyond hierarchical flows, agents can collaborate as peers. For example, iterative improvement workflows:

```python
from dataclasses import dataclass
from typing import Literal

optimizer = Agent(name="optimizer", primer="You create and hone content.")
evaluator = Agent(name="evaluator", primer="You critique and suggest improvements.")

@dataclass
class Review:
    quality: Literal["good", "average", "bad"]
    feedback: str

# Only the evaluator needs to create Review objects
evaluator.cls(Review)
optimizer.cls(Review, constructable=False)


@optimizer.task
def create_content(topic: str) -> str:  # type: ignore[return-value]
    """Create initial content on the topic."""
    pass

@optimizer.task  
def improve_content(content: str, feedback: str) -> str:  # type: ignore[return-value]
    """Improve content based on feedback."""
    pass

@evaluator.task
def review_content(content: str) -> Review:  # type: ignore[return-value]
    """Review content and provide structured feedback."""
    pass

# Iterative improvement loop with regular Python control flow
content = create_content("python decorators")
while (review := review_content(content)).quality != "good":
    content = improve_content(content, review.feedback)

print(f"Final content: {content}")
```

This peer collaboration pattern enables quality improvement, fact-checking, and iterative refinement workflows.

## 7. Event Monitoring and Debugging

One of agex's most powerful features is comprehensive event tracking that lets you see exactly what agents are thinking and doing. This is invaluable for debugging, monitoring, and understanding agent behavior.

### Basic Event Monitoring

Every agent action generates events that you can retrieve and analyze:

```python
from agex import Agent, Versioned, events

# Create agent with persistent state to capture events
agent = Agent(name="debug_agent")
state = Versioned()

@agent.task
def analyze_data(numbers: list[int]) -> dict:  # type: ignore[return-value]
    """Analyze a list of numbers and return statistics."""
    pass

# Execute the task
result = analyze_data([1, 5, 3, 9, 2, 7], state=state)
print(f"Result: {result}")

# Get all events from this agent execution
agent_events = events(state)
print(f"Generated {len(agent_events)} events")

# Events include TaskStartEvent, ActionEvent, OutputEvent, SuccessEvent, and FailEvent
# See what the agent was thinking during execution
from agex.agent.events import ActionEvent

for event in agent_events:
    if isinstance(event, ActionEvent):
        print(f"Agent was thinking: {event.thinking}")
        print(f"Agent executed: {event.code}")
```

The events system makes debugging agent behavior straightforward. For comprehensive event monitoring patterns, see the **[Events API](./api/events.md)**.

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
    print(f"Task exceeded max iterations": {e.message})
```

## Next Steps

- **[API Reference](./api/overview.md)** - Complete documentation for all agex APIs
- **[Events API](./api/events.md)** - Comprehensive guide to event monitoring and debugging
- **[Examples](./examples/overview.md)** - Real-world examples showing advanced patterns
- **[The Big Picture](./big-picture.md)** - Framework philosophy and design principles
