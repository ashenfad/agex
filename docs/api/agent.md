# Agent

The `Agent` class is the main entry point for creating AI agents in agex. Each agent manages its own set of registered capabilities (see [Registration](registration.md)) and can execute tasks (see [Task](task.md)) through a secure Python environment.

## Constructor

```python
Agent(
    primer: str | None = None,
    timeout_seconds: float = 5.0,
    max_iterations: int = 10,
    max_tokens: int = 2**16,
    name: str | None = None,
    llm_client: LLMClient | None = None,
    llm_max_retries: int = 2,
    llm_retry_backoff: float = 0.25,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primer` | `str | None` | `None` | Instructions that guide the agent's behavior and personality |
| `timeout_seconds` | `float` | `5.0` | Maximum time in seconds for task execution |
| `max_iterations` | `int` | `10` | Maximum number of think-act cycles per task |
| `max_tokens` | `int` | `65536` | Maximum tokens for context rendering |
| `name` | `str | None` | `None` | Unique identifier for the agent (auto-generated if not provided) |
| `llm_client` | `LLMClient | None` | `None` | An instantiated `LLMClient` for the agent to use. If `None`, a default client is created. |
| `llm_max_retries` | `int` | `2` | Number of times to retry a failed LLM completion before aborting with `LLMFail`. |
| `llm_retry_backoff` | `float` | `0.25` | Initial backoff (seconds) between retries. Backoff grows exponentially per attempt. |

### Examples

```python
from agex import Agent, connect_llm, LLMClient

# Simple agent using the default LLM (dummy provider or from env vars)
agent = Agent(primer="You are a helpful assistant.")

# Agent configured with a specific, explicitly created client
llm_client = connect_llm(provider="openai", model="gpt-4.1-nano", temperature=0.1)
expert_agent = Agent(
    primer="You are an expert data analyst.",
    llm_client=llm_client
)
```

## LLM Configuration

An agent's connection to a Large Language Model is managed by an `LLMClient` instance. There are two primary ways to configure this.

### 1. Direct Instantiation (Recommended)

The clearest and most explicit method is to create an `LLMClient` instance using the top-level `connect_llm()` factory function and pass it directly to the `Agent`'s constructor. This makes dependencies obvious and is ideal for production code and testing.

```python
from agex import connect_llm, Agent
from agex.llm.dummy_client import DummyLLMClient

# For production, create a client for a specific provider
prod_client = connect_llm(provider="openai", model="gpt-4.1-nano")
prod_agent = Agent(llm_client=prod_client)

# For testing, you can inject a dummy client
test_client = DummyLLMClient()
test_agent = Agent(llm_client=test_client)
```

### 2. Default Client (via Environment Variables)

If you do not pass an `llm_client` to the `Agent` constructor, `agex` will automatically create a default one for you by calling `connect_llm()` with no arguments. This default client is configured using environment variables.

```bash
# Example: Configure agent via environment variables
export AGEX_LLM_PROVIDER="openai"
export AGEX_LLM_MODEL="gpt-4.1-nano"
export OPENAI_API_KEY="your-key-here"
```

### 3. Using OpenAI-Compatible Endpoints (e.g., Ollama)

You can connect `agex` to any model provider that offers an OpenAI-compatible API endpoint, such as a local [Ollama](https://ollama.com/) server. This is done by specifying `provider="openai"` and passing the correct arguments to `connect_llm`.

```python
# Example for connecting to a local Ollama server
local_client = connect_llm(
    provider="openai",
    model="qwen3-coder:30b",   # The specific model served by Ollama
    base_url="http://localhost:11434/v1",
    api_key="ollama",          # Placeholder key for local services
)

local_agent = Agent(llm_client=local_client)
```

> **Note on Model Compatibility:** `agex` relies on the model's ability to follow specific function-calling or "tool use" instructions. While many models are compatible, we have specifically tested and verified that the `qwen3` family of models works effectively when served via Ollama. Performance may vary with other models. We recommend `qwen3-coder:30b`.

### 4. Advanced: Client vs. Completion Arguments

The `connect_llm` function is designed to intelligently separate two types of arguments:
-   **Client Arguments**: Used to configure the connection to the LLM provider (e.g., `api_key`, `base_url`, `timeout`).
-   **Completion Arguments**: Used to control the behavior of the model for each request (e.g., `temperature`, `top_p`, `max_tokens`).

You can pass both types of arguments directly to `connect_llm`. The underlying client for each provider (`OpenAIClient`, `AnthropicClient`, etc.) is responsible for correctly routing them.

```python
# Example with both client and completion arguments
client = connect_llm(
    provider="openai",
    model="gpt-4.1-nano",
    # --- Client Arguments ---
    api_key="sk-...",
    timeout=30.0,
    # --- Completion Arguments ---
    temperature=0.7,
    top_p=0.9,
)
```


## Properties

### `.name`
**Type:** `str`

The agent's unique identifier. If not provided in constructor, a random name is generated.

```python
agent = Agent()
print(agent.name)  # "agent_abc123" (auto-generated)

named_agent = Agent(name="my_assistant")
print(named_agent.name)  # "my_assistant"
```

### `.primer`
**Type:** `str | None`

The agent's behavioral instructions.

```python
agent = Agent(primer="You are concise and direct.")
print(agent.primer)  # "You are concise and direct."
```

### `.timeout_seconds`
**Type:** `float`

The maximum time in seconds allowed for a single block of agent-generated code to execute. This is a safety mechanism to prevent runaway code. It applies strictly to code execution time, not time spent waiting for the LLM.

### `.max_iterations`
**Type:** `int`

Maximum number of think-act cycles per task. If an agent doesn't complete a task within this limit, it will raise a `TaskTimeout`.

### `.max_tokens`
**Type:** `int`

Maximum number of tokens to use when rendering the agent's context.


## Agent Registry
agex automatically registers all agents in a global registry to enable inter-agent communication. For **testing**, use `clear_agent_registry()` to prevent cross-contamination between test cases.

```python
from agex import clear_agent_registry
import pytest

# Typical pytest usage
@pytest.fixture(autouse=True)
def clear_agents():
    clear_agent_registry()
    yield
    clear_agent_registry()
```

## Next Steps

- **Registration Methods**: See [Registration](registration.md) for `.fn()`, `.cls()`, and `.module()` methods
- **Task Definition**: See [Task](task.md) for `@agent.task` usage
- **State Management**: See [State](state.md) for `Versioned` objects and persistent agent memory
- **Debugging**: See [View](view.md) for inspecting agent capabilities and execution state
