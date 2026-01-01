# State Configuration

The `connect_state()` factory function configures agent memory and persistence. State determines whether agents remember context across task calls and how that memory is stored.

## `connect_state()` API

```python
from agex import connect_state

state_config = connect_state(
    type: Literal["versioned", "live"] = "versioned",
    storage: Literal["memory", "disk"] = "memory",
    path: str | None = None,  # Required for disk storage
    init: Callable[[], dict] | dict | None = None,  # Initialize state vars
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `str` | `"versioned"` | State type: `"versioned"` (with checkpointing) or `"live"` (in-memory only) |
| `storage` | `str` | `"memory"` | Storage backend: `"memory"` or `"disk"` |
| `path` | `str \| None` | `None` | Path for disk storage (required when `storage="disk"`) |
| `init` | `Callable \| dict \| None` | `None` | Initialize state variables on first session creation |

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

The `session` parameter isolates state between different users or conversations. You can also access state directly via [`agent.state(session)`](agent.md#statesession-str--default).

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

> [!WARNING]
> **Modal Sub-Agent Limitation**: When the parent agent uses Modal host, sub-agents cannot yet have persistent state. Sub-agents must use ephemeral state (no `state=` parameter).

### Disk Storage

Persistent storage that survives restarts:

```python
state = connect_state(type="versioned", storage="disk", path="/var/agex/state")
```

**Use for:** Production, remote execution, long-running workflows.

## State Initialization

The `init` parameter lets you populate state variables when a session is first created. This is useful for loading data that should be mutable within state (e.g., calendars, datasets, config objects).

```python
from agex import Agent, connect_state

def load_initial_data():
    """Called once per new session."""
    return {
        "calendar": load_calendar("events.ics"),
        "config": {"theme": "dark", "locale": "en"},
    }

agent = Agent(
    primer="You manage my calendar.",
    state=connect_state(
        type="versioned",
        storage="disk",
        path="/tmp/agex/calendar",
        init=load_initial_data,  # Callable - deferred until first session
    ),
)
```

**How it works:**
1. On first access to a session, `init()` is called (or dict is used directly)
2. Each key-value pair is set in state
3. A sentinel (`__agex_init__`) marks the session as initialized
4. For versioned state, a snapshot is committed
5. Subsequent calls skip init (sentinel detected)

> [!TIP]
> Use a callable for lazy initialization - it defers loading until the session is actually created, avoiding work at agent definition time.

## Features of Versioned State

### Automatic Checkpointing

Every agent iteration creates a snapshot. You can inspect historical states or revert the agent to a previous point in time.

**Inspecting History (Read-Only)**
Use `checkout()` to get a read-only view of the state at a specific commit:

```python
from agex import events, view

# Get events after a task run
all_events = events(resolved_state)

# Inspect state as it was when an action occurred
action = all_events[0]
historical = resolved_state.checkout(action.commit_hash)
print(view(historical, focus="full"))
```

**Reverting State (Destructive)**
Use `revert_to()` to move the agent's HEAD back to a previous commit. This orphans all subsequent commits (which can be cleaned up by GC).

```python
# Revert to the state after a specific successful task
success_event = all_events[-1]
resolved_state.revert_to(success_event.commit_hash)

# The agent continues from this point as if later actions never happened
```

> [!TIP]
> Use `state.initial_commit` to get the hash of the very first commit (the empty root state). This is useful for resetting the agent completely.

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

## Advanced: Direct State Construction

> [!NOTE]
> Most users should use `connect_state()` for configuration. Direct construction is for advanced use cases requiring fine-grained control or custom storage backends.

For advanced use cases, you can construct state objects directly instead of using the factory function:

### Direct Versioned State

```python
from agex import Versioned, Live, Disk, Memory

# Equivalent to connect_state(type="versioned", storage="disk", path="/path")
state = Versioned(Disk("/path/to/storage"))

# Equivalent to connect_state(type="versioned", storage="memory")
state = Versioned(Memory())

# Equivalent to connect_state(type="live")
state = Live()
```

### Garbage Collection for Long-Running Agents

For agents that accumulate large state histories over many iterations, use `GCVersioned` to automatically prune old commits:

```python
from agex import GCVersioned, Disk

state = GCVersioned(
    Disk("/path/to/storage"),
    high_water_bytes=100 * 1024 * 1024,  # Trigger GC at 100MB
    low_water_bytes=50 * 1024 * 1024,    # Target 50MB after GC
)
```

When state size exceeds `high_water_bytes`, the oldest commits are pruned until size drops below `low_water_bytes`. This prevents unbounded growth in long-running agents.

### Custom Storage Backends

Implement the `KVStore` interface to use custom storage backends (Redis, S3, etc.):

```python
from agex.state.kv import KVStore

class RedisStore(KVStore):
    def get(self, key: str) -> bytes | None:
        """Retrieve value for key, or None if not found."""
        ...
    
    def set(self, key: str, value: bytes) -> None:
        """Store value for key."""
        ...
    
    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """Compare-and-swap: set value only if current value matches expected."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete key."""
        ...
    
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
        ...

# Use with Versioned state
from agex import Versioned
state = Versioned(RedisStore(host="localhost", port=6379))
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
