# agex: Library-Friendly Agents

**`agex`** (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to work directly with your existing libraries and codebase. It provides a sandboxed execution environment with seamless access to the Python ecosystem.

## 25-Second Demo

![agex demo gif](assets/teaser.gif)

**This works because** `agex` provides easy runtime interoperability. The agent receives and returns real `pandas.DataFrame` and `plotly.Figure` objects, not just JSON. It works directly with your libraries. **Dive deeper with the full [`agex101.ipynb`](demos/agex101.ipynb) tutorial.**

## What Makes This Different

`agex` enables workflows without the accidental complexity of frameworks that rely on JSON or isolated execution environments. The key difference is **object passing** - `agex` transparently handles the passing of complex Python objects between your code and an agent's sandboxed environment.

## Key Features

<div class="grid cards" markdown>

-   :material-code-braces: **Runtime Interoperability**

    ---

    Agents work with real Python objects like `numpy` arrays, `pandas` DataFrames, and custom classes without JSON serialization overhead.

-   :material-function: **Code-as-Action**

    ---

    Agents compose primitives into solutions, not rigid pre-built tools.

-   :material-history: **Agent Workspace Persistence**

    ---

    Git-like versioning with automatic checkpointing and time-travel debugging.

-   :material-account-group: **Multi-Agent Orchestration**

    ---

    Natural coordination through hierarchical delegation or simple Python control flow - no complex DSLs required.

</div>

## Get Started

<div class="grid cards" markdown>

-   **[📚 Quick Start Guide](quick-start.md)**

    ---

    Learn step-by-step with hands-on examples - from basic agents to multi-agent workflows.

-   **[🔭 The Big Picture](big-picture.md)**

    ---

    Understand the core philosophy and architectural principles behind agex.

-   **[📓 Notebook Demos](demo.md)**

    ---

    See complete agex workflows with code, outputs, and explanations.

-   **[📖 API Reference](api/overview.md)**

    ---

    Complete technical documentation for all agex components and methods.

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

## Project Status

!!! warning "Pre-Release"
    `agex` is a new framework in active development. While the core concepts are stabilizing, the API should be considered experimental and is subject to change.

For teams looking for a more battle-tested library built on the same "agents-that-think-in-code" philosophy, we highly recommend Hugging Face's excellent [`smolagents`](https://github.com/huggingface/smolagents) project.