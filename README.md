# agex: Library-Friendly Agents

`agex` (a portmanteau of **age**nt **ex**ecution) is a Python library for building AI agents that work directly with your code.

You define a typed Python function with `@task` and the agent fills it in. It writes sandboxed Python that calls into the modules you've whitelisted, returning real Python objects (DataFrames, Plotly figures, your Pydantic models) that flow back into your code unchanged. There's no JSON serialization at the boundary and no separate runtime to deploy: agex runs inside your application's Python process.

Because the sandbox is pure-Python AST rewriting, the same agent code runs in-process, in a subprocess, in a kernel-isolated worker, or (via Pyodide) entirely in a browser tab. [agex-studio](https://agex.studio) is the proof-of-concept: pandas, scikit-learn, plotly, and a chat agent all running client-side with no backend.

```python
import pandas as pd
from agex import Agent

agent = Agent()
agent.module(pd)

@agent.task
def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Return summary statistics for the numeric columns."""
    pass

stats = summarize(my_dataframe)  # real dict[str, float]
```

![Demo of an agex agent returning pandas DataFrames and plotly figures in an IPython REPL](docs/assets/teaser.gif)

## What you get

- **Typed function tasks** - `@task` declares the input/output contract; the agent fulfills it.
- **Curated Python environment** - whitelist exactly which modules and classes the agent can use, with per-member visibility.
- **Versioned workspace** - virtual filesystem, the agent's session memory, and event log are all kvgit-backed, with checkpoints and time-travel.
- **Multi-agent orchestration** - coordinate agents with regular Python control flow; sub-agents are just functions.
- **Flexible execution** - in-process by default; subprocess, kernel-isolated, browser (Pyodide), or remote ([Modal](https://modal.com/), HTTP) when you need them.

For a deeper dive, see the [agex101 tutorial](https://ashenfad.github.io/agex/examples/agex101/) or the [geospatial routing example](https://ashenfad.github.io/agex/examples/routing/) for multi-library integration. For a NiceGUI integration demo, see [`agex-ui`](https://github.com/ashenfad/agex-ui).

## Documentation

Complete documentation is hosted at **[ashenfad.github.io/agex](https://ashenfad.github.io/agex/)**.

Key sections:
- **[📚 Quick Start Guide](https://ashenfad.github.io/agex/quick-start/)**
- **[🔭 The Big Picture](https://ashenfad.github.io/agex/concepts/big-picture/)**
- **[💡 Examples](https://ashenfad.github.io/agex/examples/overview/)**
- **[📖 API Reference](https://ashenfad.github.io/agex/api/overview/)**

## Installation

Install agex with your preferred LLM provider:

```bash
pip install "agex[openai]"        # OpenAI models
pip install "agex[anthropic]"     # Anthropic Claude models
pip install "agex[gemini]"        # Google Gemini models

# Or with all providers
pip install "agex[all-providers]"
```

## Project Status

This is a hobby project in active development. The core concepts are stabilizing but the API should be considered experimental.

If you're looking for a more battle-tested library built on the same "agents-that-think-in-code" idea, [`smolagents`](https://github.com/huggingface/smolagents) (Hugging Face) is the closest cousin and a good choice. agex explores a different shape: an embeddable library you import into your application, with typed function contracts and a pure-Python sandbox that runs anywhere Python runs (including the browser).

## Built On

agex is composed of several focused libraries that can also be used independently:

| Library | Purpose |
|---------|---------|
| [sandtrap](https://github.com/ashenfad/sandtrap) | In-process Python sandbox via AST rewriting |
| [kvgit](https://github.com/ashenfad/kvgit) | Versioned key-value store with git-like semantics |
| [monkeyfs](https://github.com/ashenfad/monkeyfs) | Filesystem interception via monkey-patching |
| [termish](https://github.com/ashenfad/termish) | Virtual terminal with shell-like commands |
| [reprobate](https://github.com/ashenfad/reprobate) | Budget-controlled repr for Python objects |

## Contributing

Bug reports, ideas, and pull requests welcome - see [GitHub Issues](https://github.com/ashenfad/agex/issues) or the [Contributing Guide](CONTRIBUTING.md).
