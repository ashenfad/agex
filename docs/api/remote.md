# Remote Execution

Remote execution allows you to run agent tasks on separate hosts, enabling distributed architectures and centralized LLM access. The client serializes the agent and streams results back via Server-Sent Events.

## Installation

```bash
# Client-side (for @remote decorator)
pip install agex[remote]

# Server-side (for hosting)
pip install agex[server]
```

## Quick Start

### Client

```python
from agex import Agent
from agex.remote import remote

agent = Agent()

@remote(url="http://example.com:8000")
@agent.task
def analyze_data(data: list[float]) -> dict:
    """Analyze numerical data."""
    pass

# Executes remotely, streams results back
result = analyze_data(data=[1.0, 2.0, 3.0])
```

### Server

```python
from agex.server import create_app, run_server

# Create app - LLM client is reconstructed from agent config
app = create_app(state_dir="/var/agex/state")

# Run with uvicorn
run_server(app, host="0.0.0.0", port=8000)
```

> **Note:** Ensure the server environment has the appropriate API keys set
> (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`).

Or use uvicorn directly:
```bash
uvicorn myserver:app --host 0.0.0.0 --port 8000
```

---

## The `@remote` Decorator

Wraps a task function to execute on a remote host instead of locally. Both sync and async task functions are supported.

```python
@remote(
    url="http://example.com:8000",
    timeout=30.0,
    retries=3,
)
@agent.task
def my_task() -> str:
    """Task implementation provided by remote agent."""
    pass
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | — | Remote server URL (required) |
| `state` | `str \| None` | `None` | Default state URI (can be overridden per-call) |
| `timeout` | `float` | `300.0` | Request timeout in seconds |
| `retries` | `int` | `0` | Network retry attempts (connection failures only) |

### Retry Semantics

The `retries` parameter controls automatic retry behavior for **connection-level failures only**:

**What IS retried:**
- Connection refused (server not running)
- DNS resolution failures
- Network timeouts before the request reaches the server

**What is NOT retried:**
- HTTP errors (4xx, 5xx) — the request reached the server
- Server-side execution errors — task logic failed
- SSE stream interruptions — partial execution may have occurred

> [!IMPORTANT]
> This is intentional: if a request reaches the server and execution begins, retrying could cause **duplicate side effects**. The framework errs on the side of caution — if there's any ambiguity about whether execution started, it will not retry.

### Decorator Order

`@remote` must be the **outermost** decorator:

```python
# ✓ Correct
@remote(url="...")
@agent.task
def my_task(): pass

# ✗ Wrong - will not work
@agent.task
@remote(url="...")
def my_task(): pass
```

### Callbacks

Remote tasks support the same observability callbacks as local tasks:

```python
from agex import pprint_events, pprint_tokens

result = my_task(
    on_token=pprint_tokens,
    on_event=pprint_events,
)
```

Events and tokens are streamed from the server in real-time via SSE.

### State Override

Use `state` to manage multiple independent sessions on the server:

```python
# Each user gets their own persistent state
result = my_task(state="disk://user_alice")
result = my_task(state="disk://user_bob")
```

---

## Server Configuration

### `create_app()`

Factory function that creates a FastAPI application.

```python
from agex.server import create_app

app = create_app(state_dir="/var/agex/state")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_dir` | `str` | `"/var/agex/state"` | Base directory for `disk://` state URIs |

The server reconstructs the LLM client from the serialized agent configuration.
Ensure the server environment has the appropriate API keys set.

### `run_server()`

Convenience wrapper around `uvicorn.run()`:

```python
from agex.server import create_app, run_server

app = create_app()
run_server(app, host="0.0.0.0", port=8000)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/execute` | POST | Execute a serialized agent task |
| `/health` | GET | Health check (returns `{"status": "ok"}`) |

---

## State URIs

State URIs specify persistent storage for agent state on the server.

### `disk://` Scheme

```python
result = my_task(state="disk://my_session")
```

Resolves to server-side disk storage at `{state_dir}/my_session`.

**Session ID requirements:**
- Alphanumeric characters, underscores (`_`), and hyphens (`-`) only
- No path separators (path traversal is blocked)

---

## Error Handling

### `RemoteExecutionError`

Raised when the remote task fails during execution:

```python
from agex.remote import RemoteExecutionError

try:
    result = my_task()
except RemoteExecutionError as e:
    print(f"Remote error: {e}")
```

### `RemoteTimeoutError`

Raised when the request times out:

```python
from agex.remote import RemoteTimeoutError

try:
    result = my_task()
except RemoteTimeoutError:
    print("Request timed out")
```

---

## What Gets Serialized

When `@remote` executes, the following are serialized and sent to the server:

| Included | Excluded |
|----------|----------|
| Agent configuration | Live LLM client instances |
| Registered functions (via cloudpickle) | Host-specific connections |
| Policy and namespaces | Runtime objects in `_host_object_registry` |
| LLM client *configuration* | — |

The server uses the LLM configuration to reconstruct an equivalent client with its own credentials.

---

## Compatibility

Remote execution serializes agents and results via `cloudpickle`. For reliable operation:

- **Python version**: Client and server should run the same minor version (e.g., both on 3.12.x)
- **agex version**: Keep versions aligned; class definitions may change between releases
- **Dependencies**: If registered functions use libraries (e.g., pandas, numpy), both environments need compatible versions

> [!TIP]
> For production deployments, pin exact versions in both client and server `requirements.txt` to avoid serialization mismatches.

---

## Limitations

Current remote execution constraints:

- **No fan-out** - Sub-agents cannot execute on different remote hosts
- **No live objects** - Database connections, file handles, and sockets cannot cross the wire
- **No authentication** - Auth is assumed to be handled at the infrastructure layer (e.g., API gateway, VPN)
- **disk:// only** - Only `disk://` state URIs are supported; `redis://`, `s3://` planned for future

