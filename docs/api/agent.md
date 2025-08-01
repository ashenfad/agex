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
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_client: "LLMClient | None" = None,
    **llm_kwargs
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
| `llm_provider` | `str | None` | `None` | LLM provider override (e.g., "openai"). Used if `llm_client` is not provided. |
| `llm_model` | `str | None` | `None` | LLM model override (e.g., "gpt-4"). Used if `llm_client` is not provided. |
| `llm_client` | `LLMClient | None` | `None` | An instantiated LLMClient for the agent to use. Overrides all other LLM configuration. |
| `**llm_kwargs` | | | Additional LLM parameters (temperature, etc.). Used if `llm_client` is not provided. |

### Examples

```python
from agex import Agent
from agex.llm.dummy_client import DummyLLMClient

# Basic agent
agent = Agent(primer="You are a helpful math assistant.")

# Advanced configuration with provider strings
agent = Agent(
    primer="You are an expert data analyst.",
    timeout_seconds=30.0,
    max_iterations=15,
    name="data_analyst",
    llm_model="gpt-4",
    temperature=0.1
)

# Advanced configuration with a pre-configured client (preferred)
from agex.llm import get_llm_client
my_llm_client = get_llm_client(provider="openai", model="gpt-4", temperature=0.1)
agent = Agent(
    primer="You are an expert data analyst.",
    llm_client=my_llm_client
)
```

## LLM Configuration

There are two primary ways to configure the LLM an agent uses: either by passing an instantiated `LLMClient` (preferred), or by passing configuration strings that the agent will use to create a client.

### 1. By `LLMClient` Instance (Preferred)
This is the recommended approach for flexibility and testability. You create an `LLMClient` instance first and pass it to the agent.

```python
from agex import Agent
from agex.llm import get_llm_client, DummyLLMClient

# For production
prod_client = get_llm_client(provider="openai", model="gpt-4", temperature=0.1)
prod_agent = Agent(primer="You are a production-ready assistant.", llm_client=prod_client)

# For testing
test_client = DummyLLMClient(...)
test_agent = Agent(primer="You are a test assistant.", llm_client=test_client)
```

**Benefits:**
- **Flexibility:** Use custom `LLMClient` subclasses for unsupported providers or to add custom logic (e.g., caching, logging).
- **Resource Management:** Share a single client instance (with its network connections and authentication) across multiple agents.
- **Testability:** Easily inject mock or dummy clients for testing without relying on environment variables.

### 2. By Configuration Strings
If no `llm_client` is provided, the agent will create one for you based on a fallback chain of configuration sources.

#### Global Configuration with `configure_llm()`
```python
from agex import configure_llm

configure_llm(provider="openai", model="gpt-4", temperature=0.7)
```
Sets global LLM defaults that all agents will use.

#### Environment Variables
Provider and model can also be set via environment variables:
```bash
export AGEX_LLM_PROVIDER=openai
export AGEX_LLM_MODEL=gpt-4
```

#### Per-Agent Constructor Arguments
You can override global or environment settings by passing `llm_provider` and `llm_model` strings to the `Agent` constructor.

#### Configuration Priority
1. **`llm_client` instance** (highest priority - overrides all other settings)
2. Agent constructor parameters (`llm_provider`, `llm_model`, etc.)
3. `configure_llm()` settings
4. Environment variables (lowest priority)

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

## Multi-Agent Usage

### Unique Agent Names
When creating multiple agents, ensure they have unique names for proper identification:
```python
research_agent = Agent(name="researcher", primer="You excel at research.")
writer_agent = Agent(name="writer", primer="You are a skilled writer.")
```

### Agent Registry
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
