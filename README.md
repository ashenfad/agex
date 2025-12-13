# agex: Library-Friendly Agents

**`agex`** (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to work directly with your existing libraries and codebase.

## Core Concepts

`agex` executes sandboxed Python directly in your process, bypassing JSON serialization to let complex objects flow freely. You define a safe, focused environment by whitelisting exactly which capabilities are available.

In agex, agents are just Python functions:
- **Defined by Signature**: Input/output types are enforced by standard type hints.
- **Powered by Code**: Agents write and execute Python in a secure sandbox to fulfill that signature.
- **Curated Scope**: You whitelist exactly which modules and classes are available.
- **Stateful Memory**: The entire workspace is versioned, enabling time-travel debugging and serverless-style background execution.
- **Unified Observability**: Complete visibility into agent thought and action with real-time event & token streaming.
- **Multi-Agent Orchestration**: Coordinate hierarchical or peer agents with natural Python control flow.
- **Integrated Benchmarking**: A built-in framework for data-driven agent evaluation.

![agex demo gif](docs/assets/teaser.gif)

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

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on our development workflow, code style, and how to submit pull requests. For bug reports and feature requests, please use [GitHub Issues](https://github.com/ashenfad/agex/issues).
