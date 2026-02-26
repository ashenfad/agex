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
    max_memory_mb: int | None = None,
    max_open_files: int | None = None,
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
| `max_memory_mb` | `int \| None` | `None` | Per-task memory limit in MB. Enforced by sandtrap (kernel on Linux, checkpoint on macOS). See [Resource Limits](#resource-limits). |
| `max_open_files` | `int \| None` | `None` | Maximum file descriptors per task (Unix only). See [Resource Limits](#resource-limits). |

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

Each `connect_*` parameter has its own reference page: **[LLM](llm.md)**, **[State](state.md)**, **[Host](host.md)**, **[FileSystem](fs.md)**.

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

## Class Methods

### `Agent.clone_registrations(source, *, name=None, **kwargs)`

Creates a new agent with copied registrations (modules, functions, classes) but independent state/fs/host.

```python
from agex import Agent, connect_state

sandbox = Agent.clone_registrations(
    main_agent,
    name="sandbox",
    state=connect_state(type="versioned", storage="memory"),
)
sandbox.module(ui)  # Doesn't affect main_agent
```

Takes a `source` agent (required) plus any [Agent constructor](#constructor) parameter to override.

**Returns:** A new `Agent` with copied registrations and independent state/fs/host.

---

## Standalone Functions

### `run_file_in_sandbox(agent, file_path, session, **kwargs)`

Runs a file from VFS in the agent's sandbox.

```python
from agex import run_file_in_sandbox

state = run_file_in_sandbox(sandbox, "app/main.py", session_id)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Agent` | Required | The agent providing the execution context |
| `file_path` | `str` | Required | Path to the file in VFS |
| `session` | `str` | `"default"` | Session identifier for state/fs access |
| `eval_timeout_seconds` | `float \| None` | `None` | Optional timeout override |

**Returns:** The `State` after execution.

**Raises:** `FileNotFoundError` if the file doesn't exist, `EvalError` if execution fails.

---

## Instance Methods

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

## Resource Limits

Protect against runaway agent code with per-task resource limits:

```python
agent = Agent(
    max_memory_mb=500,    # Allow 500MB additional memory per task
    max_open_files=256,   # Allow up to 256 file descriptors
)
```

### Memory Limits

Enforced by [sandtrap](https://github.com/ashenfad/sandtrap). On Linux, kernel-enforced via `RLIMIT_AS`. On macOS, checkpoint-based (loop iterations, function entries). Exceeding the limit raises `MemoryError` (wrapped in `EvalError`).

### File Descriptor Limits

Limits open file descriptors via `RLIMIT_NOFILE` (Unix). Exceeding raises `OSError`.

### Platform Support

| Platform | Memory limits | File descriptor limits |
|----------|--------------|----------------------|
| Linux | Kernel-enforced (`RLIMIT_AS`) + checkpoint-based | `RLIMIT_NOFILE` |
| macOS | Checkpoint-based only | `RLIMIT_NOFILE` |
| Windows | No-op | No-op |

### Process-Level Behavior

Memory and file descriptor limits are process-wide on Unix. For concurrent tasks in the same process, the limit applies to all tasks combined. For stronger isolation guarantees with per-task limits, use the Modal integration which provides containerized execution.

### VFS Size Limits

For virtual filesystem size limits, see [FileSystem Configuration](fs.md#size-limits).

## Network Access Control

By default, agent code cannot make network connections. This prevents data exfiltration and unauthorized API calls. To allow specific functions to use the network, register them with `network_access=True`:

```python
@agent.fn(network_access=True)
def fetch_data(url: str) -> dict:
    """Fetch data from a URL."""
    import requests
    return requests.get(url).json()
```

See **[Security - Network Access Control](../concepts/security.md#network-access-control)** for details.

## Next Steps

- **[LLM Configuration](llm.md)**: Providers, models, and API options
- **[State Configuration](state.md)**: Memory, persistence, and sessions
- **[Host Configuration](host.md)**: Local and remote execution
- **[FileSystem Configuration](fs.md)**: Virtual filesystem and file events
- **[Registration](registration.md)**: Expose capabilities with `.fn()`, `.cls()`, `.module()`
- **[Task](task.md)**: Define agent tasks with `@agent.task`
