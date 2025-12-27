# State Configuration

The `connect_state()` factory function configures agent memory and persistence. State determines whether agents remember context across task calls and how that memory is stored.

## `connect_state()` API

```python
from agex import connect_state

state_config = connect_state(
    type: Literal["versioned", "live"] = "versioned",
    storage: Literal["memory", "disk"] = "memory",
    path: str | None = None,  # Required for disk storage
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `str` | `"versioned"` | State type: `"versioned"` (with checkpointing) or `"live"` (in-memory only) |
| `storage` | `str` | `"memory"` | Storage backend: `"memory"` or `"disk"` |
| `path` | `str \| None` | `None` | Path for disk storage (required when `storage="disk"`) |

## State Types

### No State (Default)

When no state is configured, each task call is independent:

```python
agent = Agent(primer="You are helpful.")

@agent.task
def analyze(data: str) -> dict:
    """Analyze data."""
    pass

# Each call starts fresh - no memory
result1 = analyze("first")
result2 = analyze("second")  # No memory of first call
```

### Versioned State (Recommended for Persistence)

Provides checkpointing, rollback, and cross-process persistence:

```python
from agex import Agent, connect_state

agent = Agent(
    primer="You are helpful.",
    state=connect_state(type="versioned", storage="memory"),
)

@agent.task
def build_analysis(data: str) -> dict:
    """Build cumulative analysis."""
    pass

# Agent remembers context across calls
result1 = build_analysis("first")
result2 = build_analysis("second")  # Remembers first call
```

### Live State (In-Process Only)

For unpicklable objects like database connections:

```python
agent = Agent(
    primer="You are a database expert.",
    state=connect_state(type="live"),
)

# Agent can work with file handles, cursors, etc.
# Memory lost when process ends
```

## Session Management

The `session` parameter isolates state between different users or conversations:

```python
agent = Agent(
    state=connect_state(type="versioned", storage="memory"),
)

@agent.task
def chat(message: str) -> str:
    """Chat with the user."""
    pass

# Different sessions have isolated memory
chat("Hello", session="user_alice")  # Alice's conversation
chat("Hello", session="user_bob")    # Bob's conversation (separate)

# Same session shares memory
chat("Remember this", session="alice")
chat("What did I say?", session="alice")  # Remembers previous message
```

### Default Session

If you don't specify a session, the default session `"default"` is used:

```python
# These are equivalent
chat("Hello")
chat("Hello", session="default")
```

## Storage Options

### Memory Storage (Default)

Fast, in-process storage. Lost when process ends:

```python
state = connect_state(type="versioned", storage="memory")
```

**Use for:** Development, testing, single-process applications.

> [!NOTE]
> On Modal, `memory` storage uses Modal Dict (not in-process memory) with a 7-day TTL on inactive keys. Dict names are auto-generated from the agent's fingerprint. See [Host - Modal](host.md#modalserverless-execution) for details.

### Disk Storage

Persistent storage that survives restarts:

```python
state = connect_state(type="versioned", storage="disk", path="/var/agex/state")
```

**Use for:** Production, remote execution, long-running workflows.

## Features of Versioned State

### Automatic Checkpointing

Every agent iteration creates a snapshot. You can inspect or rollback to any point:

```python
from agex import events, view

# Get events after a task run
all_events = events(resolved_state)

# Each event has a commit_hash for time-travel debugging
action = all_events[0]
historical = resolved_state.checkout(action.commit_hash)
print(view(historical, focus="full"))
```

### Concurrent Task Handling

Versioned state handles concurrent execution safely via the `on_conflict` parameter on tasks:

```python
@agent.task(on_conflict="retry")  # Default: retry on conflict
def interactive_task(query: str) -> str:
    pass

@agent.task(on_conflict="abandon")  # Background: abandon on conflict
def background_task() -> None:
    pass
```

See [Task - Concurrency Control](task.md#concurrency-control) for details.

### Unpicklable Objects

Versioned state handles unpicklable objects gracefully. Agents can use database cursors, file handles, etc. - they work for single-turn use. Accessing them in later turns shows a clear error with solutions.

## Advanced: Direct State Objects

For advanced use cases, you can create state objects directly:

```python
from agex import Versioned, Live, Disk

# Direct Versioned with disk backend
state = Versioned(Disk("/path/to/storage"))

# With garbage collection for long-running agents
from agex import GCVersioned
state = GCVersioned(
    Disk("/path/to/storage"),
    high_water_bytes=100 * 1024 * 1024,  # 100MB
)
```

### Custom Storage Backends

Implement the `KVStore` interface for custom backends:

```python
from agex.state.kv import KVStore

class RedisStore(KVStore):
    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes) -> None: ...
    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool: ...
    # ... other abstract methods
```

## Quick Reference

| Configuration | Memory | Persistence | Use Case |
|--------------|--------|-------------|----------|
| No state | None | None | Simple, one-off tasks |
| `connect_state(type="live")` | In-process | None | Unpicklable objects |
| `connect_state(type="versioned", storage="memory")` | In-process | Checkpoints | Development, testing |
| `connect_state(type="versioned", storage="disk", path="...")` | Disk | Full | Production |

## Next Steps

- **[Agent](agent.md)**: Configure agents with state
- **[Task](task.md)**: Session parameter and concurrency control
- **[Host](host.md)**: State requirements for remote execution
