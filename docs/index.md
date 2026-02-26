# agex: Library-Friendly Agents

**`agex`** (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to work directly with your existing libraries and codebase.

![agex demo gif](assets/teaser.gif)

**This works because** `agex` agents can accept and return complex types like `pandas.DataFrame` and `plotly.Figure` objects without intermediate JSON serialization. **Dive deeper with the full [`agex101.ipynb`](examples/agex101.ipynb) tutorial.**

## What Makes This Different

`agex` uses a subset of Python as the agent action space, executing actions in a sandboxed environment within your process. This approach avoids the complexity of JSON serialization and allows complex objects to flow directly between your code and the agent. You control exactly what functions, classes, and modules are available, creating a powerful yet secure environment.

<div class="grid cards" markdown>

-   :material-code-braces: **Code-as-Action**

    ---

    AST-based sandbox allows agents to take action through code but within limits.

-   :material-history: **Agent Workspace Persistence**

    ---

    Agents & their compute environments persist w/ git-like versioning.

-   :material-function: **Library Integration**

    ---

    Agents integrate with existing Python libraries (not restricted to tools).


-   :material-account-group: **Multi-Agent Orchestration**

    ---

    Hierarchical agents or agent coordination via Python control flow.

-   :material-camera-iris: **Event Streams**

    ---
    
    Observe agents in real-time with notebook-friendly event streaming.

-   :material-chart-bar: **Benchmarking**

    ---

    Test & evaluate agent performance with data-driven metrics.

</div>

## Get Started

<div class="grid cards" markdown>

-   **[📚 Quick Start Guide](quick-start.md)**

    ---

    Learn step-by-step with hands-on examples - from basic agents to multi-agent workflows.

-   **[🔭 The Big Picture](concepts/big-picture.md)**

    ---

    Understand the core philosophy and architectural principles behind agex.

-   **[:material-flask-outline: Code Examples](examples/overview.md)**

    ---
    
    Explore advanced patterns and core concepts with runnable Python examples.


-   **[📖 API Reference](api/overview.md)**

    ---

    Complete technical documentation for all agex components and methods.

-   **[:material-code-tags-check: Demo App: `agex-ui`](https://github.com/ashenfad/agex-ui)**

    ---
    
    See an agent build a dynamic web UI on the fly with NiceGUI. A live example of agent-driven development.

</div>

## Installation

Install agex with your preferred LLM provider:

=== "OpenAI"

    ```bash
    pip install "agex[openai]"
    ```

=== "Anthropic"

    ```bash
    pip install "agex[anthropic]"
    ```

=== "Gemini"

    ```bash
    pip install "agex[gemini]"
    ```

=== "All Providers"

    ```bash
    pip install "agex[all-providers]"
    ```

## Built On

agex is composed of several focused libraries that can also be used independently:

- [**sandtrap**](https://github.com/ashenfad/sandtrap) — In-process Python sandbox via AST rewriting
- [**kvgit**](https://github.com/ashenfad/kvgit) — Versioned key-value store with git-like semantics
- [**monkeyfs**](https://github.com/ashenfad/monkeyfs) — Filesystem interception via monkey-patching
- [**termish**](https://github.com/ashenfad/termish) — Virtual terminal with shell-like commands
- [**reprobate**](https://github.com/ashenfad/reprobrate) — Budget-controlled repr for Python objects

## Project Status

!!! warning "Alpha"
    `agex` is a new framework in active development. While the core concepts are stabilizing, the API should be considered experimental and is subject to change.

For teams looking for a more battle-tested library built on the same "agents-that-think-in-code" philosophy, we highly recommend Hugging Face's excellent [`smolagents`](https://github.com/huggingface/smolagents) project. `agex` explores a different architectural path, focusing on deep runtime interoperability and a secure, sandboxed environment for direct integration with existing Python libraries.