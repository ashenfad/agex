# agex: Library-Friendly Agents

**`agex`** (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to work directly with your existing libraries and codebase.

## Core Concepts

`agex` executes sandboxed Python directly in your process, bypassing JSON serialization to let complex objects flow freely. You define a safe, focused environment by whitelisting exactly which capabilities are available.

Key features:
- **Type-Safe Execution**: Agents fulfill typed signatures by executing sandboxed Python.
- **Curated Scope**: Whitelist exactly which modules and classes are available.
- **Versioned Workspace**: A virtual filesystem, an explicit per-session `cache`, and the event log are all kvgit-backed for time-travel debugging.
- **Multi-Agent Orchestration**: Coordinate agents with natural Python control flow.
- **Terminal & Scripting**: Agents run shell commands, Python scripts, and git — all sandboxed.
- **Flexible Hosting**: Run locally (default), on HTTP servers, or serverless via [Modal](https://modal.com/).

![Demo of an agex agent returning pandas DataFrames and plotly figures in an IPython REPL](docs/assets/teaser.gif)

**This works because** `agex` agents can accept and return complex types like `pandas.DataFrame` and `plotly.Figure` objects without intermediate JSON serialization. For a deeper dive, check out the full **[agex101.ipynb tutorial](https://ashenfad.github.io/agex/examples/agex101/)** or see **[geospatial routing with OSMnx](https://ashenfad.github.io/agex/examples/routing/)** for advanced multi-library integration.

For a full demo app where agex integrates with NiceGUI, see [`agex-ui`](https://github.com/ashenfad/agex-ui).


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
# Install with a specific provider
pip install "agex[openai]"        # For OpenAI models
pip install "agex[anthropic]"     # For Anthropic Claude models
pip install "agex[gemini]"        # For Google Gemini models

# Or install with all providers
pip install "agex[all-providers]"
```

## Project Status

> **⚠️** `agex` is a new framework in active development. While the core concepts are stabilizing, the API should be considered experimental and is subject to change.

For teams looking for a more battle-tested library built on the same "agents-that-think-in-code" philosophy, we highly recommend Hugging Face's excellent [`smolagents`](https://github.com/huggingface/smolagents) project. `agex` explores a different architectural path, focusing on deep runtime interoperability and a secure, sandboxed environment for direct integration with existing Python libraries.

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

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on our development workflow, code style, and how to submit pull requests. For bug reports and feature requests, please use [GitHub Issues](https://github.com/ashenfad/agex/issues).
