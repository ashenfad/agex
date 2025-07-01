# agex API Reference

Welcome to the agex API documentation. agex is a Python agentic framework that enables LLM agents to work with real Python objects through runtime interoperability.

## Core APIs

### [Agent](agent.md)
Create and configure agents with LLM providers, timeouts, and custom primers.

### [Registration](registration.md) 
Register functions, classes, and modules to make them available to agents. Control visibility and configure per-member settings.

### [Task](task.md)
Define agent tasks using the `@agent.task` decorator. Support for standalone tasks and multi-agent workflows.

### [State](state.md)
Manage persistent state across agent executions with automatic checkpointing and rollback capabilities.

### [View](view.md) ⚠️ *Experimental*
Inspect agents and their execution state for debugging. API subject to frequent changes.

## Quick Start

```python
from agex import Agent, Versioned

# Create an agent
agent = Agent()

# Register a function
@agent.fn
def calculate_sum(a: int, b: int) -> int:
    return a + b

# Define a task
@agent.task("Add two numbers and explain the result")
def math_explainer(x: int, y: int, state: Versioned) -> str:
    pass  # type: ignore[return-value]

# Execute with state
state = Versioned()
result = math_explainer(5, 3, state=state)
print(result)
```

## Import Patterns

Most agex functionality is available at the top level:

```python
from agex import Agent, Versioned, view
from agex import Memory, Disk, Cache  # Storage backends
from agex import configure_llm, clear_agent_registry  # Utilities
```

## Framework Status

agex is pre-0.0.1 and under active development. APIs may change as the framework evolves toward its first release.

For examples and higher-level documentation, see the main [docs/](../) directory. 