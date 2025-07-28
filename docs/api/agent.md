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
| `llm_provider` | `str | None` | `None` | LLM provider override (falls back to environment config) |
| `llm_model` | `str | None` | `None` | LLM model override (falls back to environment config) |
| `**llm_kwargs` | | | Additional LLM parameters (temperature, etc.) |

### Examples

```python
from agex import Agent

# Basic agent
agent = Agent(primer="You are a helpful math assistant.")

# Advanced configuration
agent = Agent(
    primer="You are an expert data analyst.",
    timeout_seconds=30.0,
    max_iterations=15,
    name="data_analyst",
    llm_model="gpt-4",
    temperature=0.1
)

# Agent with no behavioral instructions (uses only system primer)
agent = Agent(name="generic_helper")
```

## LLM Configuration

### Global Configuration with `configure_llm()`

```python
from agex import configure_llm

configure_llm(provider="openai", model="gpt-4", temperature=0.7)
```

Sets global LLM defaults that all agents will use unless overridden in their constructor.

**Parameters:**
- `provider`: LLM provider ("openai", etc.)
- `model`: Model name (e.g., "gpt-4", "gpt-3.5-turbo")  
- `**kwargs`: Additional parameters like `temperature`, `max_tokens`, etc.

### Environment Variables

Provider and model can also be set via environment variables:

```bash
export AGEX_LLM_PROVIDER=openai
export AGEX_LLM_MODEL=gpt-4
```

**Configuration Priority:**
1. Agent constructor parameters (highest priority)
2. `configure_llm()` settings
3. Environment variables (lowest priority)

**Usage:**
```python
# Set global defaults
configure_llm(provider="openai", model="gpt-4", temperature=0.1)

# All agents use these defaults
agent1 = Agent(primer="You are helpful.")
agent2 = Agent(primer="You are concise.")

# Override for specific agent
agent3 = Agent(primer="You are creative.", temperature=0.9)
```

### Per-Agent Configuration

Individual agents can override global settings via constructor parameters:

```python
# Global setting
configure_llm(model="gpt-3.5-turbo")

# This agent uses gpt-4 instead
special_agent = Agent(
    primer="You need advanced reasoning.",
    llm_model="gpt-4",
    temperature=0.0
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

The agent's behavioral instructions. These are included in the system message to guide the agent's personality and approach.

```python
agent = Agent(primer="You are concise and direct.")
print(agent.primer)  # "You are concise and direct."
```

### `.timeout_seconds`
**Type:** `float`

The maximum time in seconds allowed for a **single block of agent-generated code to execute**.

This is a critical safety mechanism designed to prevent runaway code, such as an infinite `while True:` loop. If a single code evaluation cycle exceeds this duration, the execution is stopped, and an `EvalError` is injected into the agent's environment with a descriptive message. The agent can see this error, understand that its previous code was too slow or stuck, and attempt a different approach.

**Scope of the Timeout:**

*   **It applies strictly to code execution.** Time spent waiting for an LLM provider to respond with a new action does **not** count towards this timeout.
*   **It is per-evaluation.** In multi-agent workflows, time spent by a sub-agent executing its task does not count against the parent agent's timeout for a separate evaluation cycle. Each agent's execution is timed independently.

### `.max_iterations`
**Type:** `int`

Maximum number of think-act cycles per task. If an agent doesn't complete a task within this limit, it will raise a `TaskTimeout`.

### `.max_tokens`
**Type:** `int`

Maximum number of tokens to use when rendering the agent's context (recent state, available functions, etc.). This controls how much information the agent can see at once.

## Multi-Agent Usage

### Unique Agent Names

When creating multiple agents, ensure they have unique names for proper identification:

```python
research_agent = Agent(name="researcher", primer="You excel at research.")
writer_agent = Agent(name="writer", primer="You are a skilled writer.")

# Names must be unique within the same application
assert research_agent.name != writer_agent.name
```

### Agent Registry

agex automatically registers all agents in a global registry to enable inter-agent communication. Each agent gets a unique fingerprint:

```python
agent = Agent(name="my_agent")
print(agent.fingerprint)  # UUID string
```

For **testing**, use `clear_agent_registry()` to prevent cross-contamination between test cases:

```python
from agex import clear_agent_registry

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