# Host Configuration

The `connect_host()` factory function configures where agent tasks execute. By default, agents run locally. For distributed deployments, you can run agents on remote servers.

## `connect_host()` API

```python
from agex import connect_host

host = connect_host(
    provider: Literal["local", "http", "modal"] = "local",
    **kwargs,  # Host-specific arguments
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"local"` | Host provider: `"local"`, `"http"`, or `"modal"` |
| `url` | `str` | | (HTTP only) Server URL, e.g., `"http://localhost:8000"` |
| `secrets` | `str \| list[str]` | | (Modal only, **required**) Modal secret names for API keys |
| `app` | `str` | Auto-generated | (Modal only) Modal app name |
| `gpu` | `str` | `None` | (Modal only) GPU type (e.g., `"A10G"`, `"T4"`, `"A100"`) |
| `memory` | `int` | `None` | (Modal only) Memory in MB |
| `timeout` | `float` | `300.0` | Execution timeout in seconds |
| `scaledown_window` | `int` | `300` | (Modal only) Seconds before idle containers scale down |

## Remote Execution Model

When using remote hosts (HTTP or Modal), **the entire agent ReAct loop executes on the remote server** - not just the final generated code. This includes:

- **LLM calls**: All reasoning and code generation happens server-side
- **Sandbox evaluation**: Generated code executes in the remote environment
- **State updates**: State mutations happen using server-accessible storage
- **Event streaming**: Events and tokens stream back to the client in real-time

**Why this matters:**
- LLM latency is measured from the server's network location
- Compute resources (CPU, GPU, memory) are allocated on the remote host
- API keys and secrets must be configured server-side
- Only the final result and streaming events are sent back to the client

This architecture enables GPU-accelerated tasks, distributed compute, and cost-effective serverless execution.

## Host Types

### Local Execution (Default)

Tasks run in the current Python process:

```python
from agex import Agent

# Implicit local execution
agent = Agent(primer="You are helpful.")
```

### Modal/Serverless Execution

Tasks run on Modal's serverless infrastructure with automatic scaling and GPU support:

```python
from agex import Agent, connect_host, connect_state

host = connect_host(
    provider="modal",
    secrets="llm-keys",       # Required: Modal secret names
    gpu="A10G",              # Optional: GPU type
    memory=4096,             # Optional: Memory in MB
    timeout=600.0,           # Optional: Execution timeout
    scaledown_window=300,    # Optional: Idle seconds before scale-down
)

# Memory storage: Dict with 7-day TTL (auto-named from agent fingerprint)
state = connect_state(type="versioned", storage="memory")

# OR Disk storage: Dict + Volume (forever, requires path for Volume naming)
state = connect_state(type="versioned", storage="disk", path="my-agent")

agent = Agent(
    primer="You are helpful.",
    host=host,
    state=state,
)
```

**Key Features:**
- **Auto-deploy**: First task execution automatically builds and deploys the Modal function
- **GPU support**: Specify GPU types for compute-intensive tasks
- **Two storage tiers**:
  - `memory` → Disk cache + Modal Dict (7-day TTL on inactive keys)
  - `disk` → Disk cache + Modal Dict + Modal Volume (forever)
- **Dependency inference**: Automatically detects and installs required packages

**Execution:**
```python
# First call auto-deploys and executes
result = my_task("hello", session="user-123")
```

**Optional warmup** (pre-builds the image for faster first request):
```python
agent.warmup()  # Builds container image and deploys to Modal
```

**State Requirements:**

Modal host supports two storage modes:

```python
# ✅ Memory: Uses Disk + Modal Dict (7-day TTL on inactive keys)
# Auto-named from agent fingerprint if no path provided
state = connect_state(type="versioned", storage="memory")

# ✅ Disk: Uses Disk + Modal Dict + Volume (forever)
# Requires path to name the Volume
state = connect_state(type="versioned", storage="disk", path="my-agent")

# ✅ Ephemeral: Fresh state per invocation
state = connect_state(type="ephemeral")

# ❌ Live state: Not supported (no persistence between invocations)
state = connect_state(type="live", storage="disk")  # Raises ValueError
```

> [!NOTE]
> **Auto-naming**: With `storage="memory"`, Dict names are auto-generated from the agent's fingerprint (e.g., `agex.a1b2c3d4.default`). With `storage="disk"`, the `path` parameter is used to name both the Dict and Volume.

> [!WARNING]
> **7-day TTL**: Modal Dict entries expire after 7 days of inactivity. Reads refresh the TTL, so active sessions persist indefinitely. Use `storage="disk"` for truly permanent state.

> [!WARNING]
> **Modal Sub-Agent Limitation**: Sub-agents with Modal hosts cannot currently be registered as capabilities on parent agents. This prevents double-serialization overhead. This limitation may be relaxed in the future to allow Modal containers to spawn new Modal functions. For now, use HTTP hosts for explicit distributed multi-agent workflows.

### HTTP/Remote Execution

Tasks run on a remote server:

```python
from agex import Agent, connect_host, connect_state

host = connect_host(provider="http", url="http://agent-server:8000")
state = connect_state(type="versioned", storage="disk", path="/shared/state")

agent = Agent(
    primer="You are helpful.",
    host=host,
    state=state,  # Must use disk storage for remote
)
```

## Remote Execution Architecture

When using HTTP host:

1. **Agent serialization**: The agent (registrations, LLM config) is serialized
2. **Server execution**: Task runs on the remote server
3. **State persistence**: State is resolved server-side from the path
4. **Event streaming**: Events stream back via SSE

### Server Setup

Start the agex server:

```bash
uvicorn agex.server:app --host 0.0.0.0 --port 8000
```

Or with a custom state directory:

```python
from agex.server import create_app

app = create_app(state_dir="/var/agex/state")
```

### State Requirements

Remote execution requires disk-based state with a shared path:

```python
# ✅ Works: Disk storage with explicit path
state = connect_state(type="versioned", storage="disk", path="/shared/state")

# ❌ Fails: Memory storage has no shared path
state = connect_state(type="versioned", storage="memory")
```

## Hierarchical Agents and Host Inheritance

In multi-agent workflows, resource inheritance follows these rules:

| Resource | Inheritance | Notes |
|----------|-------------|-------|
| **LLM** | Independent | Each agent uses its own LLM (or default) |
| **Host** | Independent | Sub-agents default to Local (run in-process) |
| **State** | Independent | Each agent uses its own state config |
| **Session** | Inherited | Session ID passes from parent to sub-agents |

### Example: Remote Orchestrator with Local Sub-Agents

```python
# Orchestrator runs remotely
orchestrator = Agent(
    name="orchestrator",
    host=connect_host(provider="http", url="http://server:8000"),
    state=connect_state(type="versioned", storage="disk", path="/state"),
)

# Sub-agent has no explicit host → defaults to Local
# When orchestrator calls sub-agent, it runs locally ON THE SERVER
specialist = Agent(name="specialist")

@orchestrator.fn
@specialist.task
def process_data(data: str) -> str:
    """Specialist task."""
    pass
```

When `orchestrator` calls `process_data`:
1. Orchestrator's task runs on remote server A
2. Orchestrator's code invokes `process_data`
3. Specialist's host is rehydrated from its config
4. If specialist has Local host → runs locally on server A
5. If specialist has HTTP host → makes its own HTTP call
6. Session is inherited, each agent resolves its own state

> [!TIP]
> **Sub-agents can run on different remote hosts.** When a sub-agent has its own HTTP host configured, it makes a separate HTTP call and runs on that server. This enables GPU offloading where a CPU orchestrator delegates compute-intensive work to GPU-equipped servers.

## Callbacks with Remote Execution

Event and token callbacks work with remote hosts:

```python
from agex import pprint_events, pprint_tokens

@agent.task
def my_task(query: str) -> str:
    """Process query."""
    pass

# Callbacks receive events as they stream from the server
result = my_task("hello", on_event=pprint_events, on_token=pprint_tokens)
```

## Limitations

- **Serialization**: Agent registrations must be serializable (standard library + common packages)
- **State storage**: Remote hosts require disk-based state with accessible paths
- **Network**: Events stream via SSE; plan for network latency

## Next Steps

- **[Agent](agent.md)**: Configure agents with hosts
- **[State](state.md)**: Configure state for remote execution
- **[LLM](llm.md)**: Configure LLM providers
