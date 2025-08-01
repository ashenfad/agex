# agex API Reference

Welcome to the agex API documentation. agex is a Python agentic framework that enables LLM agents to work with real Python objects through runtime interoperability.

**New to agex?** Start with the **[Quick Start Guide](../quick-start.md)** for hands-on examples and step-by-step learning. This API reference provides complete technical documentation for all agex components.

## Core APIs

- **[Agent](agent.md)** -
Create and configure agents with LLM providers, timeouts, and custom primers.

- **[Registration](registration.md)** -
Register functions, classes, and modules to make them available to agents. Control visibility and configure per-member settings.

- **[Task](task.md)** -
Define agent tasks using the `@agent.task` decorator. Support for standalone tasks and multi-agent workflows.

- **[State](state.md)** -
Manage persistent state across agent executions with automatic checkpointing and rollback capabilities.

- **[Events](events.md)** -
Access comprehensive agent execution events for debugging, monitoring, and multi-agent coordination. Real-time introspection into agent thinking and actions.

- **[Error Handling](errors.md)** -
Understand agent task control functions and exception handling (`TaskFail`, `TaskClarify`, `TaskTimeout`).

- **[View](view.md)** -
Inspect agents and their execution state for debugging.

- **[Helpers](helpers.md)** -
Convenience functions for registering common libraries like NumPy and pandas.

- **[Benchmarking](../benchmarking.md)** -
Empirical evaluation framework for agent performance testing, A/B testing primers, and regression detection.

## Import Patterns

Most agex functionality is available at the top level:

```python
from agex import Agent, Versioned, view
from agex import Memory, Disk, Cache  # Storage backends
from agex import TaskFail  # Error handling
from agex import clear_agent_registry  # Utilities
```

## API Design Philosophy

### Definable vs. Configurable

The agex API is built on a core principle that makes its syntax consistent and predictable:

- **Definable** methods are for registering new functions (`def`) and classes (`class`) as they are being defined. These methods (`.fn`, `.cls`) support the decorator pattern (`@agent.fn`) and use a "decorator factory" syntax.

- **Configurable** methods are for exposing parts of *already existing* objects, like imported modules. These methods (`.module`) are direct function calls: `agent.module(math, ...)`.

This distinction means **the syntax itself tells you what kind of operation you're performing**:

```python
# Definable - decorator pattern for new code
@agent.fn
def my_new_function():
    pass

@agent.cls
class MyNewClass:
    pass

# Configurable - direct calls for existing code  
agent.module(math, include=["sin", "cos"])
agent.module(pandas, visibility="low")
```

### Registration Override Principle

**More specific registrations override more general ones.** This enables powerful workflows where you can bulk-register with low visibility, then "promote" specific important members:

```python
# Bulk register with low visibility
agent.module(numpy, visibility="low")

# Promote specific functions to high visibility
agent.fn(numpy.array, visibility="high")
agent.fn(numpy.mean, visibility="high")
```

This layered approach gives you both broad capability exposure and fine-grained control over what agents see prominently.

### Visibility and Context Management

The three-tier visibility system (`high`/`medium`/`low`) addresses the core challenge of LLM context management:

- **`high`**: Shows full signatures and documentation - for core capabilities
- **`medium`**: Shows signatures only - for supporting functions  
- **`low`**: Available but hidden - for broad library access without clutter

This allows agents to have access to extensive capabilities while keeping their context focused on the most relevant tools.

## Framework Status

!!! warning "Pre-Release"
    `agex` is a new framework in active development. While the core concepts are stabilizing, the API should be considered experimental and is subject to change.

For examples and higher-level documentation, see the main [docs/index.md](../index.md).
