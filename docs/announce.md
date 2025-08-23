# Meet Agex

Most agentic frameworks require you to wrap your code in tool abstractions and deal with JSON serialization. To avoid that I built `agex`—a Python-native agentic framework where agents work directly with your existing libraries and runtime.

Its closest relative is Hugging Face's excellent `smol-agents`. While both "think-in-code", `agex` focuses on interoperability, allowing agents to receive and return complex Python objects like DataFrames, Plotly figures, or even callables.

For example:

```python
import math
from typing import Callable
from agex import Agent

agent = Agent(primer="You are an expert at writing small, useful functions.")

# Equip the agent with the math module
agent.module(math)

# The fn sig is the contract; the agent provides the implementation at runtime
@agent.task
def build_function(prompt: str) -> Callable:
    """Build a callable function from a text prompt."""
    pass

# The agent returns a real, callable Python function, not a JSON blob
is_prime = build_function("a function that checks if a number is prime")

# You can use it immediately
print(f"Is 13 prime? {is_prime(13)}")
# > Is 13 prime? True
```

It works by parsing agent-generated code into an AST and running it in a sandbox allowing only whitelisted operations. Since the sandbox is in your runtime, it eases the flow of complex objects between your code and the agent.

From the agent's point-of-view, it lives in a Python REPL. It has its own stdout with which to inspect data and see errors in order to self-correct when completing tasks. An agent's REPL is persisted across tasks, so agents can build their own helpers and improve over time.

**Highlights:**

- **Code-as-Action**: AST-style Python sandbox for agent actions.
- **Library Integration**: Plug into libraries rather than tools (& skip the JSON).
- **Workspace Persistence**: Git-like versioning for agent state (w/ time-travel debugging).
- **Multi-Agent**: Orchestrate agents with natural Python control flow.
- **Observability**: Real-time, notebook-friendly event streams.
- **Benchmarking**: A framework for data-driven agent evaluation.

One demo notebook worth highlighting is an agent that uses `OSMnx` and `Folium` libraries for geospatial routing. When faced with a novel constraint ("avoid this highway"), it doesn't fail; it writes its own helper on the fly to solve the problem. You can see that here: [Interactive Routing Demo](https://ashenfad.github.io/agex/examples/routing/).

The project is pre-release but the core concepts are stabilizing. I'm hoping to find a few brave souls to kick the tires. Thanks!

- **GitHub:** [https://github.com/ashenfad/agex](https://github.com/ashenfad/agex)
- **Docs & Examples:** [https://ashenfad.github.io/agex/](https://ashenfad.github.io/agex/)
