# agex: Library-Friendly Agents

**`agex`** (a portmanteau of **age**nt **ex**ecution) is a Python-native agentic framework that enables AI agents to work directly with your existing libraries and codebase. It provides a sandboxed execution environment with seamless access to the Python ecosystem.

## 30-Second Example

```python
import math
from typing import Callable
from agex import Agent

agent = Agent()
agent.module(math)  # Share capabilities with the agent 

@agent.task
def make_function(description: str) -> Callable:  # type: ignore[return-value]
    """Generate a Python function from a text description."""
    pass  # Empty body - the agent implements this function

# Agent returns an actual Python callable you can use immediately
prime_finder = make_function("find the next prime larger than a given number")

print(prime_finder(100))  # 101
my_data.sort(key=prime_finder)  # Works with existing Python code
```

**This works because** `agex` provides easy runtime interoperability - agents don't just return JSON, they create real Python objects that live directly in your runtime environment.

## What Makes This Different

`agex` enables workflows without the accidental complexity of frameworks that rely on JSON or isolated execution environments. The key difference is **object passing** - `agex` transparently handles the passing of complex Python objects between your code and an agent's sandboxed environment.

## Key Features

<div class="grid cards" markdown>

-   :material-code-braces: **Runtime Interoperability**

    ---

    Agents work with real Python objects like `numpy` arrays, `pandas` DataFrames, and custom classes without JSON serialization overhead.

-   :material-account-group: **Multi-Agent Orchestration**

    ---

    Simple hierarchical agent workflows using standard Python control flow. No YAML or complex DSLs required.

-   :material-eye: **Comprehensive Events**

    ---

    Complete visibility into agent behavior with time-travel debugging and execution introspection.

-   :material-shield-check: **Secure Sandbox**

    ---

    Whitelist-based execution environment with AST-level validation and controlled capability exposure.

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