# agex

`agex` provides a Python emulation environment, enabling agents to 'think in code'.

## Building an Agent

Since a `agex` agent thinks in code, the agent definition is a micro-Python DSL. We'll
create an instance of an agent and it will have decorators available to mark functions
and classes we want the agent to access, along with with tasks the agent will do.

For `agex` agents, a tool is simply a `fn` it can use. And an agent `task` is a function
where the agent decides how to implement it.

```python
import math
import agex

math_agent = agex.Agent(primer="You are a helpful math assistant.")

@math_agent.fn  # expose a fn to an agent via decorator
def sqrt(num: float) -> float:
    """Calculate the square root of a number"""
    return num ** 0.5

math_agent.fn(math.sin)  # expose a fn via functional registration

@math_agent.task  # mark a fn def as an agent task, implementation is the agent's job
def assist(prompt: str) -> str:
    """Assist the user with their math questions"""
    pass
```

Use `cls` decorator to give an agent access to classes. You may expose fns, classes, or even entire modules. You may whitelist/blacklist module fns (or class methods) and decide
whether to let the agent know about them or thier documentation within the agent context.

Finally, multi-agent coordination may be done by dual-decoration. The fn signature provides the `task` entry point for one agent while the `fn` decorator makes that agent
callable for the other. Since each agent has different perspectives, you'll likely want
to include documentation directly in the decorators rather than the fn docstring. The
function signature encapsulates the contract between the agents.

```python
math_agent = agex.Agent(primer="...")
orchestrator = agex.Agent(primer="...")

@orchestrator.fn("A math-oriented assistant")
@math_agent.task("Assist the user with their math questions")
def math_expert(prompt: str) -> str:
    pass
```

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