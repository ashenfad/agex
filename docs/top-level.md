# Top-Level API Design (`@agent.task`)

This document outlines the design for the high-level `@agent.task` decorator, which provides a developer-friendly way to define and interact with agent-driven tasks.

## Core Concepts

The primary goal is to make defining an agent task feel as natural as defining a Python function. The developer should only need to provide a function signature and a docstring, and the agent should handle the implementation.

### Decorator Behavior

1.  **Function Replacement:** The `@agent.task` decorator does not merely wrap the decorated function; it completely **replaces** it.
2.  **Implementation Constraint:** The body of a function decorated with `@agent.task` must be empty (i.e., contain only `pass`). The decorator will raise an exception if it finds any other statements. The agent, not the developer, is responsible for the implementation.
3.  **Signature and Docstring:** The decorator's main purpose is to capture the original function's definition:
    *   Its name.
    *   Its argument signature (parameter names, types, defaults).
    *   Its return type annotation.
    *   Its docstring, which will serve as the prompt or goal for the agent.
4.  **Agent Loop Trigger:** The new function that replaces the original one will be responsible for triggering the agent's internal "think-act" loop to produce a result that matches the requested return type.
5.  **State Management:** The replacement function will have a new, optional parameter added to its signature, typically `state: State | None = None`.
    *   If a `State` object is provided, the agent will use it for long-term memory across multiple calls.
    *   If no `State` object is provided (the default), the agent will operate in a single-shot, ephemeral mode for that call.

## Developer Experience Example

From the developer's perspective, defining and using an agent task should be as simple as:

```python
from tic import Agent

# 1. Instantiate the agent
my_agent = Agent()

# 2. Define a task with a clear signature and docstring
@my_agent.task
def generate_random_number(min_val: int, max_val: int) -> int:
    """
    Generates a random integer between min_val and max_val, inclusive.
    """
    pass

# 3. Use the agent-powered function directly
# (Agent runs in single-shot mode)
random_num = generate_random_number(1, 100)
print(f"The agent generated: {random_num}")

# Or, use it with persistent state
from tic.state import Ephemeral
agent_state = Ephemeral()
another_num = generate_random_number(1, 100, state=agent_state)

```

This design abstracts away the complexity of the evaluation loop, state management, and prompting, providing a clean and intuitive API. 