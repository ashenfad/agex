# Host Configuration

The `connect_host()` factory function configures where agent tasks execute. By default, agents run locally. For distributed deployments, you can run agents on remote servers.

## `connect_host()` API

```python
from agex import connect_host

host = connect_host(
    provider: Literal["local", "http"] = "local",
    **kwargs,  # Host-specific arguments
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"local"` | Host provider: `"local"` or `"http"` |
| `url` | `str` | | (HTTP only) Server URL, e.g., `"http://localhost:8000"` |

## Host Types

### Local Execution (Default)

Tasks run in the current Python process:

```python
from agex import Agent

# Implicit local execution
agent = Agent(primer="You are helpful.")
```

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
