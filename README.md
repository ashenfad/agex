# agex

`agex` (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to think, act, and create using real Python code. It provides a secure, sandboxed execution environment that allows for deep interoperability with your existing codebase, moving beyond simple tool-calling to true runtime integration.

## 30-Second Example

```python
import math
from typing import Callable
from agex import Agent

agent = Agent()
agent.module(math)  # Give the agent math tools

@agent.task
def make_function(description: str) -> Callable:  # type: ignore[return-value]
    """Generate a Python function from a text description."""
    pass  # Empty body - the agent implements this function

# Agent returns an actual Python callable you can use immediately
prime_finder = make_function("find the next prime larger than a given number")

print(prime_finder(100))  # 101
my_data.sort(key=prime_finder)  # Works with existing Python code
```

**This works because** `agex` provides true runtime interoperability - agents don't just return JSON, they create real Python objects that live directly in your runtime environment.

**Which enables** seamless data flow between agents (numpy arrays, pandas DataFrames), hierarchical agent orchestration through normal Python control flow, and dynamic code generation that integrates immediately with existing systems.

[Learn more about the core philosophy in The Big Picture](./docs/big-picture.md).

## What Makes This Different

`agex` enables workflows that require significantly more complexity and boilerplate in frameworks that rely on JSON or isolated execution environments. Here's how the core capabilities work:

### 1. Seamless Runtime Interoperability

`agex` transparently handles the passing of complex Python objects between your code and an agent's sandboxed environment. You can work with rich objects like `numpy` arrays, `pandas` DataFrames, and custom classes without extra work—types that would be difficult or impossible to handle in frameworks limited to JSON.

For example, in [`examples/viz.py`](./examples/viz.py), one agent generates bulk `numpy.ndarray` objects, which are then passed directly to another agent that uses `plotly` to create a `Figure` for visualization.

### 2. Dynamic Code Generation & Extension

Agents can generate and return executable Python functions and classes at runtime. This allows them to not just use tools, but to create them.

In [`examples/funcy.py`](./examples/funcy.py), an agent is tasked with building a `Callable` function from a text prompt. The returned function is a real Python object that can be immediately integrated into existing logic (e.g., `my_list.sort(key=agent_generated_function)`).

### 3. Hierarchical Agent Orchestration

`agex` is designed for building complex systems out of specialized agents. One agent's core `task` can be exposed as a simple `fn` (tool) for another agent, enabling natural and powerful composition.

[`examples/multi.py`](./examples/multi.py) demonstrates this with an `orchestrator` agent that solves a high-level "idea" by delegating data generation and plotting tasks to two different specialist sub-agents. All orchestration is done with simple Python control flow—no YAML or complex DSLs required.

### 4. Live Object Integration

Agents can work directly with complex, stateful APIs without requiring wrapper classes or simplified interfaces. `agex` safely exposes live Python objects—including unpickleable ones like database connections—while maintaining state serialization safety.

[`examples/db.py`](./examples/db.py) showcases this with raw SQLite integration: agents work directly with `sqlite3.Connection` and `Cursor` objects, handling complex method chaining (`db.execute().fetchall()`) and transaction management. No `DatabaseManager` wrapper needed—agents adapt to the existing API.

### 5. Beyond Tools: Granular Function Execution

While many agent frameworks use the term "tool," `agex` deliberately uses **`fn`** to signify a more fundamental concept.

*   **Tools** often imply high-level, stateless operations where inputs and outputs are easily serialized to JSON. This model is practical for workflows where an LLM reasons between each action.
*   An `agex` **`fn`** can be much more granular. Because agents think in code, they can compose many low-level function calls into a complete program within a single execution step. This allows for complex and efficient problem-solving without the latency and cost of an LLM call at every step.

This distinction is key to enabling agents that don't just *use* tools, but truly *program* with them.

## Project Status

`agex` is a new framework in active development. While the core concepts are stabilizing, the API should be considered experimental and is subject to change.

For teams looking for a more battle-tested library built on the same "agents-that-think-in-code" philosophy, we highly recommend Hugging Face's excellent [`smolagents`](https://github.com/huggingface/smolagents) project. `agex` explores a different architectural path centered on a secure-by-design execution environment and deep runtime interoperability.

## Building More Complex Agents

Beyond the basic example above, `agex` agents can be equipped with tools and capabilities:

```python
import math
import agex

# Create an agent with specialized knowledge
math_agent = agex.Agent(primer="You are a helpful math assistant.")

# Give it custom tools
@math_agent.fn
def sqrt(num: float) -> float:
    """Calculate the square root of a number."""
    return num ** 0.5

# Expose entire modules
math_agent.module(math)

# Define what the agent should accomplish  
@math_agent.task
def assist(prompt: str) -> str:  # type: ignore[return-value]
    """Provide assistance for mathematical questions and problems."""
    pass

# The agent uses all available tools to complete the task
result = assist("What is the sin of pi/2?")
```

**Key concepts:**
- **`@agent.fn`**: Register custom functions as tools
- **`agent.module()`**: Expose existing Python modules  
- **`@agent.task`**: Define what you want accomplished (empty body - agent implements it)

## API Documentation

Complete API reference documentation is available:

**📖 [API Reference](./docs/api/overview.md)** - Complete documentation for all agex APIs

Key sections:
- **[Agent](./docs/api/agent.md)** - Creating and configuring agents
- **[Registration](./docs/api/registration.md)** - Exposing functions, classes, and modules to agents  
- **[Task](./docs/api/task.md)** - Defining agent tasks with `@agent.task`
- **[State](./docs/api/state.md)** - Persistent memory and state management
- **[View](./docs/api/view.md)** - Inspecting agents and execution state *(experimental)*

For design concepts and higher-level documentation, see:
- **[The Big Picture](./docs/big-picture.md)** - Framework philosophy, architecture, and multi-agent patterns
- **[Security Model](./docs/security.md)** - Execution environment and safety guarantees

## Setup

This project uses `pyenv` to manage the Python version and `uv` for package management.

1.  **Set up the Python version:**
    If you have `pyenv` installed, it will automatically pick up the version from the `.python-version` file.

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    uv pip install -e ".[dev]"
    ```

3.  **Set up pre-commit hooks:**
    ```bash
    pre-commit install
    ```