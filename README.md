# agex

`agex` (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to think, act, and create using real Python code. It provides a secure, sandboxed execution environment that allows for deep interoperability with your existing codebase, moving beyond simple tool-calling to true runtime integration.

The core philosophy of `agex` is that **code is the most powerful language for formal reasoning**. Instead of inventing new paradigms, `agex` provides agents with a familiar, REPL-like environment equipped with tools developers have used for decades: state inspection, introspection (`dir()`, `help()`), and modular design.

[Learn more about the core philosophy in The Big Picture](./docs/big-picture.md).

## Key Features

`agex` enables workflows that are difficult or impossible in frameworks that rely on JSON or isolated execution environments.

### 1. Seamless Runtime Interoperability

`agex` transparently handles the passing of complex Python objects between your code and an agent's sandboxed environment. By leveraging a robust serialization strategy (based on `pickle`), `agex` can work with rich objects like `numpy` arrays, `pandas` DataFrames, and custom classes—types that would be difficult or impossible to handle in frameworks limited to JSON. This pass-by-value approach ensures data integrity and prevents unintended side effects.

For example, in [`examples/viz.py`](./examples/viz.py), one agent generates bulk `numpy.ndarray` objects, which are then passed directly to another agent that uses `plotly` to create a `Figure` for visualization.

### 2. Dynamic Code Generation & Extension

Agents can generate and return executable Python functions and classes at runtime. This allows them to not just use tools, but to create them.

In [`examples/funcy.py`](./examples/funcy.py), an agent is tasked with building a `Callable` function from a text prompt. The returned function is a real Python object that can be immediately integrated into existing logic (e.g., `my_list.sort(key=agent_generated_function)`).

### 3. Hierarchical Agent Orchestration

`agex` is designed for building complex systems out of specialized agents. One agent's core `task` can be exposed as a simple `fn` (tool) for another agent, enabling natural and powerful composition.

[`examples/multi.py`](./examples/multi.py) demonstrates this with an `orchestrator` agent that solves a high-level "idea" by delegating data generation and plotting tasks to two different specialist sub-agents. All orchestration is done with simple Python control flow—no YAML or complex DSLs required.

### 4. Beyond Tools: Granular Function Execution

While many agent frameworks use the term "tool," `agex` deliberately uses **`fn`** to signify a more fundamental concept.

*   **Tools** often imply high-level, stateless operations where inputs and outputs are easily serialized to JSON. This model is practical for workflows where an LLM reasons between each action.
*   An `agex` **`fn`** can be much more granular. Because agents think in code, they can compose many low-level function calls into a complete program within a single execution step. This allows for complex and efficient problem-solving without the latency and cost of an LLM call at every step.

This distinction is key to enabling agents that don't just *use* tools, but truly *program* with them.

## Project Status

`agex` is a new framework in active development. While the core concepts are stabilizing, the API should be considered experimental and is subject to change.

For teams looking for a more battle-tested library built on the same "agents-that-think-in-code" philosophy, we highly recommend Hugging Face's excellent [`smolagents`](https://github.com/huggingface/smolagents) project. `agex` explores a different architectural path centered on a secure-by-design execution environment and deep runtime interoperability.

## Quick Start: Building an Agent

An `agex` agent is defined through a micro-Python DSL. You create an `Agent` instance and use its methods to expose capabilities (`fn`, `cls`, `module`) or define high-level goals (`task`).

*   A **`fn`** is a tool the agent can use.
*   A **`task`** is a high-level goal. You define the function signature (the "what"), and the agent provides the implementation (the "how"). This is why the function body must be empty (`pass`).

```python
import math
import agex

# 1. Create an agent with a purpose
math_agent = agex.Agent(primer="You are a helpful math assistant.")

# 2. Give it tools
@math_agent.fn
def sqrt(num: float) -> float:
    """Calculate the square root of a number."""
    return num ** 0.5

# Expose an entire module's functionality
math_agent.module(math)

# 3. Give it a task to accomplish
@math_agent.task
def assist(prompt: str) -> str:
    """
    Assistance for mathematical questions and problems.
    """
    pass

# 4. Run the task
# result = assist("What is the sin of pi/2?")
```

For a detailed guide to the API, including class and module registration, visibility controls, and multi-agent patterns, please see the [Top-Level API Design](./docs/top-level.md). All agent code is executed within a secure, whitelisted environment. You can learn more about the [Security Model here](./docs/security.md).

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