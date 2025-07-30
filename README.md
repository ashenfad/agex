# agex

`agex` (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to work directly with your existing libraries and codebase. It provides a sandboxed execution environment enabling easy access to your Python ecosystem. Libraries can integrate directly without intermediate tooling logic.

## 25-Second Demo

![agex demo gif](docs/assets/teaser.gif)

**This works because** `agex` provides easy runtime interoperability. The agent receives and returns real `pandas.DataFrame` and `plotly.Figure` objects, not just JSON. It works directly with your libraries.

**📚 [Get a Quick Start](https://ashenfad.github.io/agex/quick-start/)** - Learn step-by-step with hands-on examples

**🔭 [Get a Big Picture](https://ashenfad.github.io/agex/big-picture/)** - Learn about the core philosophy

## What Makes This Different

`agex` enables workflows without the accidental complexity of frameworks that rely on JSON or isolated execution environments. The key difference is **object passing** - `agex` transparently handles the passing of complex Python objects between your code and an agent's sandboxed environment, working with rich objects like `numpy` arrays, `pandas` DataFrames, and custom classes without extra work.

Some aspects of this approach:

### **Dynamic Code Generation**

As shown earlier, agents can generate and return executable Python functions and classes at runtime.

In [`examples/funcy.py`](./examples/funcy.py) we show this in more
detail as an agent is tasked with building a `Callable` function from a text prompt. The returned function is a real Python fn that can be immediately integrated into existing logic (e.g., `my_list.sort(key=agent_made_func)`).

### **Fns vs Tools**

Most frameworks require you to anticipate how users will combine capabilities by bundling low-level operations into high-level "tools." This leads to either an explosion of specific tools or inefficient, multi-step interactions.

**The Traditional Approach: Inflexible Tools**

Imagine giving an agent a few simple statistical tools:
```python
@tool
def calculate_mean(data: list[float]) -> float:
  """Calculates the mean of a list of numbers."""
  ...

@tool
def calculate_median(data: list[float]) -> float:
  """Calculates the median of a list of numbers."""
  ...
```
If a user asks for "the mean and the median," the agent must make **two separate tool calls**, increasing latency and cost. To be efficient, you would need to write a *new*, very specific tool that does both. This is not scalable.

**The `agex` Approach: Composing Primitives**

With `agex`, you just provide the fundamental building blocks. The agent itself writes the code to combine them efficiently for any request.

```python
import statistics
from agex import Agent

# Create an agent and give it access to the primitives.
agent = Agent()
agent.module(statistics)

@agent.task
def analyze(data: list[float], request: str) -> dict: # type: ignore[return-value]
    """Analyzes the data to fulfill the user's request."""
    pass

# The agent can now handle a more complex, multi-step request in a single pass.
my_data = [1, 2, 3, 4, 5, 6, 100]
result = analyze(
    my_data, 
    "What are the mean and median for only the positive numbers?"
)

# The agent generated code to filter the list, then call
# statistics.mean() and statistics.median(), returning the
# result in one shot.
print(result) # {'mean': 17.28, 'median': 4}
```

Because agents think in code, they can compose many low-level function calls into a complete program within a single execution step. This shifts the work of writing composite operations from you to the agent. For a deeper dive on how this code-native approach compares to industry tooling standards, see our [note on MCP and the `agex` philosophy](./docs/agex-and-mcp.md).

See [`examples/mathy.py`](./examples/mathy.py) for agents handling complex mathematical transformations without custom tools.

See [`examples/viz.py`](./examples/viz.py) for agents producing and ingesting large datasets without wrappers or JSON serialization.

### **Live Object Integration**

Agents can work directly with complex, stateful APIs without requiring wrapper classes. `agex` exposes live Python objects—including unpickleable ones like database connections—while maintaining state serialization safety.

```python
# connect to a database and share the instance methods to the agent
conn = sqlite3.connect(...)  
agent.module(conn, name="db", include=["execute", "commit"])
```

[`examples/db.py`](./examples/db.py) showcases this with raw SQLite integration: agents work directly with `sqlite3.Connection` and `Cursor` objects. No `DatabaseManager` wrapper needed—agents adapt to the existing API.

### **Agent Orchestration**

`agex` supports complex systems of specialized agents. One agent's core `task` can be exposed as a simple `fn` (tool) for another agent. The function definition becomes the contract between agents. Complex data shapes can flow directly between the agents through these shared functions.

Examples of multi-agent patterns:

- **[`examples/hierarchical.py`](./examples/hierarchical.py)**: An `orchestrator` agent delegates data generation and plotting tasks to specialist sub-agents to solve high-level visualization ideas
- **[`examples/evaluator_optimizer.py`](./examples/evaluator_optimizer.py)**: Iterative improvement through agent collaboration, where one agent creates content and another critiques it until quality criteria are met—all orchestrated with a simple Python `while` loop

All orchestration is done with simple Python control flow—no YAML or complex DSLs required.

### **Agents Architecting Agents**

As the ultimate test of library-friendliness, agents can use the `agex` API itself to design and spawn other agents at runtime. This demonstrates how naturally agents can integrate with any Python library—even `agex`.

```python
architect = Agent(name="architect", primer=PRIMER)
architect.cls(Agent)  # Let the architect use agex!
```

**Why this matters:** While a bit mind-bending, the real takeaway is how seamlessly agents can work with your existing code. See [`examples/dogfood.py`](./examples/dogfood.py) for the complete implementation.

## Project Status

`agex` is a new framework in active development. While the core concepts are stabilizing, the API should be considered experimental and is subject to change.

For teams looking for a more battle-tested library built on the same "agents-that-think-in-code" philosophy, we highly recommend Hugging Face's excellent [`smolagents`](https://github.com/huggingface/smolagents) project. `agex` explores a different architectural path centered on a secure-by-design execution environment and deep runtime interoperability.



## Documentation

Complete documentation is hosted at **[ashenfad.github.io/agex](https://ashenfad.github.io/agex/)**

**📖 [API Reference](https://ashenfad.github.io/agex/api/overview/)** - Complete technical documentation for all agex APIs

Key sections:
- **[Agent](https://ashenfad.github.io/agex/api/agent/)** - Creating and configuring agents
- **[Registration](https://ashenfad.github.io/agex/api/registration/)** - Exposing functions, classes, and modules to agents  
- **[Task](https://ashenfad.github.io/agex/api/task/)** - Defining and executing agent tasks
- **[State](https://ashenfad.github.io/agex/api/state/)** - Persistent memory with a git-like history that lets you `checkout` the agent's workspace at any point in time.
- **[Events](https://ashenfad.github.io/agex/api/events/)** - A complete event log where every agent action is linked to a versioned state snapshot for powerful time-travel debugging.
- **[View](https://ashenfad.github.io/agex/api/view/)** - Inspecting agents and execution state *(experimental)*

For design concepts and higher-level documentation, see:
- **[The Big Picture](https://ashenfad.github.io/agex/big-picture/)** - Framework philosophy, architecture, and multi-agent patterns
- **[Security Model](https://ashenfad.github.io/agex/security/)** - Execution environment and safety guarantees
- **[Sandbox Limitations](https://ashenfad.github.io/agex/development/nearly-python/)** - Understanding agent code constraints and Python compatibility

## Installation

Install agex with your preferred LLM provider:

```bash
# Install with specific provider
pip install "agex[openai]"        # For OpenAI models
pip install "agex[anthropic]"     # For Anthropic Claude models  
pip install "agex[gemini]"        # For Google Gemini models

# Or install with all providers
pip install "agex[all-providers]"
```

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup and workflow
- Code style guidelines  
- Testing requirements
- How to submit pull requests

For bug reports and feature requests, please use [GitHub Issues](https://github.com/ashenfad/agex/issues).

