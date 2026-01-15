# Agent

The `Agent` class is the main entry point for creating AI agents in agex. Each agent manages its own set of registered capabilities (see [Registration](registration.md)) and can execute tasks (see [Task](task.md)) through a secure Python environment.

## Constructor

```python
Agent(
    primer: str | None = None,
    eval_timeout_seconds: float = 5.0,
    max_iterations: int = 10,
    name: str | None = None,
    capabilities_primer: str | None = None,
    llm: LLM | None = None,
    llm_max_retries: int = 2,
    host: Host | None = None,
    state: StateConfig | None = None,
    fs: FSConfig | None = None,
    log_high_water_tokens: int | None = None,
    log_low_water_tokens: int | None = None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primer` | `str \| None` | `None` | Instructions that guide the agent's behavior and personality |
| `eval_timeout_seconds` | `float` | `5.0` | Maximum time in seconds for agent-generated code execution |
| `max_iterations` | `int` | `10` | Maximum number of think-act cycles per task |
| `name` | `str \| None` | `None` | Unique identifier for the agent (auto-generated if not provided) |
| `capabilities_primer` | `str \| None` | `None` | Optional curated text that replaces the default capabilities listing |
| `llm` | `LLM \| None` | `None` | LLM client from `connect_llm()`. If `None`, uses environment defaults. See [LLM Configuration](llm.md). |
| `llm_max_retries` | `int` | `2` | Number of times to retry a failed LLM completion |
| `host` | `Host \| None` | `None` | Execution host from `connect_host()`. If `None`, runs locally. See [Host Configuration](host.md). |
| `state` | `StateConfig \| None` | `None` | State config from `connect_state()`. If `None`, tasks are stateless. See [State Configuration](state.md). |
| `fs` | `FSConfig \| None` | `connect_fs(type="virtual")` | FileSystem config from `connect_fs()`. Defaults to an in-memory Virtual Filesystem (VFS). Pass `None` to disable. See [FileSystem Configuration](fs.md). |
| `log_high_water_tokens` | `int \| None` | `None` | Trigger event log summarization when tokens exceed this threshold |
| `log_low_water_tokens` | `int \| None` | `None` | Target token count after summarization (defaults to 50% of high water) |

### Basic Example

```python
from agex import Agent, connect_llm, connect_state

agent = Agent(
    name="analyst",
    primer="You are an expert data analyst.",
    llm=connect_llm(provider="openai", model="gpt-4.1-nano"),
    state=connect_state(type="versioned", storage="memory"),
)

@agent.task
def analyze(data: str) -> dict:
    """Analyze the provided data."""
    pass

result = analyze("sales data for Q1")
```

## Configuration

### LLM Configuration

Configure how the agent communicates with language models:

```python
from agex import Agent, connect_llm

# Explicit provider and model
llm = connect_llm(provider="anthropic", model="claude-haiku-4-5")
agent = Agent(llm=llm)

# Or use environment defaults
agent = Agent()  # Uses AGEX_LLM_PROVIDER, AGEX_LLM_MODEL env vars
```

See **[LLM Configuration](llm.md)** for providers, timeouts, and advanced options.

### State Configuration

Configure agent memory and persistence:

```python
from agex import Agent, connect_state

# Versioned state with in-memory storage
agent = Agent(
    state=connect_state(type="versioned", storage="memory"),
)

# Persistent disk storage
agent = Agent(
    state=connect_state(type="versioned", storage="disk", path="/var/agex/state"),
)
```

See **[State Configuration](state.md)** for sessions, storage backends, and advanced options.

### Host Configuration

Configure where agent tasks execute:

```python
from agex import Agent, connect_host, connect_state

# Local execution (default)
agent = Agent()

# Remote execution
agent = Agent(
    host=connect_host(provider="http", url="http://agent-server:8000"),
    state=connect_state(type="versioned", storage="disk", path="/shared/state"),
)
```

See **[Host Configuration](host.md)** for remote execution and distributed deployments.

### FileSystem Configuration

Configure secure, state-backed filesystem access:

```python
from agex import Agent, connect_fs

# Enable virtual filesystem
agent = Agent(
    state=connect_state(type="versioned", storage="disk", path="/tmp/state"),
    fs=connect_fs(type="virtual"),
)
```

See **[FileSystem Configuration](fs.md)** for virtual filesystems, file uploads, and events.

## Properties

### `.name`
**Type:** `str`

The agent's unique identifier. Auto-generated if not provided.

```python
agent = Agent()
print(agent.name)  # "agent_abc123" (auto-generated)

named_agent = Agent(name="my_assistant")
print(named_agent.name)  # "my_assistant"
```

### `.primer`
**Type:** `str | None`

The agent's behavioral instructions.

### `.eval_timeout_seconds`
**Type:** `float`

Maximum time for a single block of agent-generated code to execute (*not* LLM call time).

### `.max_iterations`
**Type:** `int`

Maximum think-act cycles per task. Raises `TaskTimeout` if exceeded.

## Methods

### `.state(session: str = "default")`

Returns the agent's state object for a given session. This is useful for:
- Inspecting state with `view(state)` (see [View API](view.md))
- Reading event history with `events(state)` (see [Events API](events.md))
- Task cancellation (see [Task Cancellation](task.md#task-cancellation))

```python
from agex import Agent, connect_state, view, events

agent = Agent(
    state=connect_state(type="versioned", storage="disk", path="/tmp/state"),
)

@agent.task
def my_task() -> str:
    """Do something."""
    pass

my_task()

# Inspect state
state = agent.state()  # Default session
print(view(state))

# Get events
for event in events(state):
    print(event)

# Specific session
state = agent.state(session="user_123")
```

**Host compatibility:**

| Host | Access |
|------|--------|
| Local | Full access |
| HTTP | ❌ Not supported (state is on remote server) |
| Modal | Full access |


## Capabilities Primer

By default, the agent's system message includes capabilities rendered from registrations. You can override this with curated text:

```python
from agex import summarize_capabilities

# Generate a curated primer
primer_text = summarize_capabilities(agent, target_chars=8000)
agent.capabilities_primer = primer_text
```

### `summarize_capabilities()` Helper

```python
def summarize_capabilities(
    agent: Agent,
    target_chars: int,
    llm: LLM | None = None,
    use_cache: bool = True,
) -> str
```

Generates a concise, guidance-oriented primer from the agent's registrations. Results are cached at `.agex/primer_cache/`.

## Event Log Summarization

For long-running agents, automatic summarization keeps the context window manageable:

```python
agent = Agent(
    log_high_water_tokens=20000,  # Trigger at 20k tokens
    log_low_water_tokens=10000,   # Target 10k after summarization
)
```

**How it works:**
1. Before each LLM call, checks total event tokens
2. If exceeding high water, summarizes oldest events
3. Replaces old events with a `SummaryEvent`

See [Events - SummaryEvent](events.md#summaryevent) for details.

## Agent Registry

agex registers all agents in a global registry for inter-agent communication. For testing, clear the registry between tests:

```python
from agex import clear_agent_registry
import pytest

@pytest.fixture(autouse=True)
def clear_agents():
    clear_agent_registry()
    yield
    clear_agent_registry()
```

## Advanced: Custom System Instructions

Override the built-in agex primer for specialized architectures:

```python
# WARNING: Removes all built-in safety rails
agent = Agent(
    agex_primer_override="Custom instructions for specialized agent..."
)
```

Use for A/B testing system prompts or model-specific optimizations.

## Next Steps

- **[LLM Configuration](llm.md)**: Providers, models, and API options
- **[State Configuration](state.md)**: Memory, persistence, and sessions
- **[Host Configuration](host.md)**: Local and remote execution
- **[FileSystem Configuration](fs.md)**: Virtual filesystem and file events
- **[Registration](registration.md)**: Expose capabilities with `.fn()`, `.cls()`, `.module()`
- **[Task](task.md)**: Define agent tasks with `@agent.task`
