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
    chaptering_trigger: int | None = None,
    max_memory_mb: int | None = None,
    max_open_files: int | None = None,
    eval_tick_limit: int | None = 100_000,
    isolation: Literal["none", "process", "kernel"] = "none",
    max_spawns: int = 8,
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
| `chaptering_trigger` | `int \| None` | `None` | Trigger [chaptering](../concepts/chapters.md) when input tokens exceed this threshold |
| `max_memory_mb` | `int \| None` | `None` | Per-task memory limit in MB. Enforced by sandtrap (kernel on Linux, checkpoint on macOS). See [Resource Limits](#resource-limits). |
| `max_open_files` | `int \| None` | `None` | Maximum file descriptors per task (Unix only). See [Resource Limits](#resource-limits). |
| `eval_tick_limit` | `int \| None` | `100_000` | Maximum Python control-flow checkpoints (loop iterations, function entries) per code execution. Set to `None` to disable. |
| `isolation` | `Literal["none", "process", "kernel"]` | `"none"` | Sandbox isolation level. See [Sandbox Isolation](#sandbox-isolation). |
| `max_spawns` | `int` | `8` | Maximum number of concurrent in-agent [spawn](../concepts/big-picture.md#what-this-enables) clones. Bounds the thread pool backing `spawn.submit` / `spawn.map`. |

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

### `.scope_names`
**Type:** `set[str]`

The capability scopes this agent *declares* across its registrations — the scopes that can be granted. Static (derived from registrations), and distinct from `scopes(state).list()`, which reports the scopes currently *granted* in a given session. See [Registration — Scoped Capabilities](registration.md#scoped-capabilities).

```python
agent.fn(send_mail, scope="email")
agent.module(requests, scope="net")
print(agent.scope_names)  # {"email", "net"}
```

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

## Chaptering (Context Compaction)

For long-running agents, chaptering keeps the context window manageable by letting agents close completed work into named chapters:

```python
agent = Agent(
    chaptering_trigger=100_000,  # Trigger chaptering above this
)
```

**How it works:**
1. After each task completes, checks `input_tokens` against the chaptering trigger
2. If exceeded, triggers the `__chapter__` task — the agent reviews its history and creates `Chapter` instances
3. The framework converts chapters to `ChapterEvent` instances, preserving originals for VFS browsing

See [Concepts: Chapters](../concepts/chapters.md) for the full explanation.

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

## Sandbox Isolation

Controls how agent-generated code is executed by [sandtrap](https://github.com/ashenfad/sandtrap):

```python
agent = Agent(isolation="process")  # Subprocess isolation
```

| Level | Description | Use Case |
|-------|-------------|----------|
| `"none"` | In-process execution. Lowest overhead. | Local notebooks, trusted agent code |
| `"process"` | Subprocess execution. Crash, OOM, or segfault in agent code won't take down the host. | Production servers, untrusted code |
| `"kernel"` | Subprocess + kernel-level restrictions (seccomp on Linux, Seatbelt on macOS). Filesystem, syscall, and network access are locked down by the OS. | High-security environments |

### What works across isolation levels

All core agent functionality works regardless of isolation level:

- **Task control**: `task_success()`, `task_fail()`, `task_clarify()` (returning normally from the Python emission is the implicit continue)
- **`print()`**: Output is captured via snapshot and returned as events
- **`view_image()`**: Image actions are collected in-sandbox and processed after execution
- **Registered functions, classes, and modules**: Survive pickle across the process boundary
- **State sync**: Namespace changes in the subprocess are synced back to state

### Pairing with filesystem

Isolation level pairs with `fs` configuration:

- `isolation="kernel"` + `connect_fs(type="isolated", root="/path")` — kernel-enforced filesystem lockdown to the IsolatedFS root
- `isolation="kernel"` + `connect_fs(type="virtual")` — no kernel filesystem lockdown, but seccomp and network isolation still apply
- `isolation="kernel"` + `fs=None` — no file I/O at all

### Hierarchical agents

When using the dual-decorator pattern (`@orchestrator.fn` + `@specialist.task`), the parent and child agents **cannot both** use process or kernel isolation. The parent's sandbox would need to fork a child for the sub-agent, but daemon processes can't create children.

The recommended pattern is `isolation="none"` on the orchestrator (its code is just function calls) and `isolation="process"` or `"kernel"` on the leaf agents that run generated code:

```python
orchestrator = Agent(isolation="none", ...)      # Coordinates
data_maker   = Agent(isolation="kernel", ...)     # Runs generated code
plotter      = Agent(isolation="kernel", ...)     # Runs generated code
```

Attempting to register an isolated sub-agent task on an isolated parent raises `ValueError` at registration time.

See **[Security - Sandbox Isolation](../concepts/security.md#sandbox-isolation)** for details.

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

Memory and file descriptor limits are process-wide on Unix. For concurrent tasks in the same process, the limit applies to all tasks combined. For per-task isolation, use `isolation="process"` or `"kernel"` (see [Sandbox Isolation](#sandbox-isolation)), or the Modal integration for containerized execution.

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
