# agex: Library-Friendly Agents

**`agex`** (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to work directly with your existing libraries and codebase.

![agex demo gif](docs/assets/teaser.gif)

**This works because** `agex` agents can accept and return complex types like `pandas.DataFrame` and `plotly.Figure` objects without intermediate JSON serialization. For a deeper dive, check out the full **[agex101.ipynb tutorial](https://ashenfad.github.io/agex/examples/agex101/)** or see **[geospatial routing with OSMnx](https://ashenfad.github.io/agex/examples/routing/)** for advanced multi-library integration.

## What Makes This Different

`agex` uses a subset of Python as the agent action space, executing actions in a sandboxed environment within your process. This approach avoids the complexity of JSON serialization and allows complex objects to flow directly between your code and the agent. You control exactly what functions, classes, and modules are available, creating a powerful yet secure environment.

-   **Code-as-Action**: AST-based sandbox allows agents to take action through code but within safe limits.
-   **Direct Library Integration**: Expose your existing code directly to agents through a registration system, bypassing the need for a separate tool abstraction layer.
-   **Workspace Persistence**: Agent compute environments persist with git-like versioning.
-   **Multi-Agent**: Build hierarchical agents or coordinate via simple Python control flow.
-   **Event Streams**: Observe your agents through notebook-friendly events.
-   **Benchmarking**: Benchmark agent outputs to understand & improve their performance.

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

> **⚠️ Pre-Release**  
> `agex` is a new framework in active development. While the core concepts are stabilizing, the API should be considered experimental and is subject to change.

For teams looking for a more battle-tested library built on the same "agents-that-think-in-code" philosophy, we highly recommend Hugging Face's excellent [`smolagents`](https://github.com/huggingface/smolagents) project. `agex` explores a different architectural path, focusing on deep runtime interoperability and a secure, sandboxed environment for direct integration with existing Python libraries.

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on our development workflow, code style, and how to submit pull requests. For bug reports and feature requests, please use [GitHub Issues](https://github.com/ashenfad/agex/issues).
