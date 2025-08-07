# State Management

Agent state in agex gives you flexible control over memory and persistence across [agent tasks](task.md). You can choose between stateless execution for maximum flexibility, in-memory state for multi-step tasks, or persistent state for production workflows.

## State Management Approaches

agex provides three ways to manage state when calling an agent task.

### 1. No State (Default)
When you call a task without a `state` parameter, the execution is **stateless**.

- **No persistence**: Variables do not survive between task calls.
- **Maximum flexibility**: Agents can work with any Python object, including unpicklable ones like database cursors or file handles.
- **Use for**: Simple, one-off tasks and prototyping.

```python
@agent.task
def analyze_data(data: list[float]) -> dict:  # type: ignore[return-value]
    """Analyze data without memory."""
    pass

# Each call is independent. No state object is needed.
result1 = analyze_data([1, 2, 3])
result2 = analyze_data([4, 5, 6])  # No memory of the previous call
```

### 2. In-Memory State (`Live`)
When you pass a `Live` state object, the agent gains **in-memory, process-bound state**.

- **Process-bound memory**: The agent remembers variables across calls within the same Python process.
- **Maximum flexibility**: Like stateless calls, this mode supports unpicklable objects.
- **No checkpointing**: Memory is lost when the process ends. There are no rollback capabilities.
- **Use for**: Multi-step workflows within a single session that need to remember stateful, unpicklable objects.

```python
from agex import Live

@agent.task  
def build_analysis(data: list[float]) -> dict:  # type: ignore[return-value]
    """Build cumulative analysis with memory."""
    pass

# State persists across calls within the same process.
shared_state = Live()
result1 = build_analysis([1, 2, 3], state=shared_state)
result2 = build_analysis([4, 5, 6], state=shared_state)  # Remembers result1
```

### 3. Persistent State (`Versioned`)
When you pass a `Versioned` state object, the agent gains **persistent, versioned state**.

- **Cross-process persistence**: The agent remembers variables across calls and process restarts (when using a storage backend).
- **Automatic checkpointing**: Every agent execution creates a commit snapshot.
- **Rollback safety**: You can debug by reverting to any previous state.
- **Constraints**: All objects in the agent's memory must be picklable.
- **Use for**: Production workflows, multi-agent coordination, and tasks requiring auditability or debugging.

```python
from agex import Versioned

@agent.task  
def build_analysis(data: list[float]) -> dict:  # type: ignore[return-value]
    """Build cumulative analysis with memory."""
    pass

# State persists across calls and can survive process restarts.
shared_state = Versioned()
result1 = build_analysis([1, 2, 3], state=shared_state)
result2 = build_analysis([4, 5, 6], state=shared_state)  # Remembers result1
```

## Features of `Versioned` State

When you use `Versioned` state, you get powerful features automatically:

- **Checkpointing**: Each agent execution creates a commit snapshot.
- **Mutation Detection**: Side-effect changes to objects are automatically captured.
- **Rollback Safety**: The framework can revert to any previous state for debugging.

### Working with Unpicklable Objects in Versioned State
A key constraint of `Versioned` state is that all stored objects must be serializable (picklable). Therefore, an agent **cannot assign unpicklable objects like database connections to variables.**

The correct pattern is to use these resources and consume their results in a single, chained operation. For a complete example, see [`db.py`](../../examples/db.py) and its associated primer, which coaches the agent on the correct `db.execute(...).fetchall()` pattern.

## Inspecting Historical State

A key feature of `Versioned` state is time-travel debugging. Every event is stamped with a `commit_hash`, allowing you to check out the agent's exact memory at that point in time.

1.  Run a task and get the events.
2.  Find an event of interest and get its `commit_hash`.
3.  Use `state.checkout()` to get a read-only view of that historical state.

```python
from agex import ActionEvent, events, view

# 1. Find an event of interest after a run
all_events = events(state)
action_event = next(e for e in all_events if isinstance(e, ActionEvent))

# 2. Get the commit hash from that event
commit_to_inspect = action_event.commit_hash

# 3. Checkout the state to a new variable
historical_state = state.checkout(commit_to_inspect)

# `historical_state` is a read-only view of the past.
print(f"--- Inspecting state at {commit_to_inspect[:7]} ---")
print(view(historical_state, focus="full"))
```

## Storage Options for `Versioned` State

You can configure where `Versioned` state is stored.

### Memory (Default)
In-memory storage that is lost when the process ends.
```python
from agex import Versioned

state = Versioned()
```
**Use for:** Development, testing, live sessions.

### Disk Storage
Persistent storage that survives process restarts.
```python
from agex import Versioned, Disk

# Path to storage directory
state = Versioned(Disk("/path/to/storage"))

# Optional size limit (default: 1GB)
state = Versioned(Disk("/path/to/storage", size_limit=500*1024*1024))
```
**Use for:** Production deployments, long-running workflows.

### Cached Disk Storage
A performant option that uses an in-memory cache on top of disk persistence.
```python
from agex import Versioned, Cache, Disk

disk_store = Disk("/path/to/storage")
state = Versioned(Cache(disk_store, max_bytes=64*1024*1024))
```
**Use for:** Performance-critical applications with large state.

## Custom Storage Backends

For distributed or specialized storage needs, you can implement the `KVStore` interface:

```python
from agex import Versioned
from agex.state.kv import KVStore

class RedisStore(KVStore):
    def __init__(self, redis_url):
        # Implementation details...
        pass
    
    def get(self, key: str) -> bytes | None:
        # Return bytes from Redis
        pass
    
    def set(self, key: str, value: bytes) -> None:
        # Store bytes in Redis
        pass
    
    # ... implement remaining abstract methods

# Use with your agent
state = Versioned(RedisStore("redis://localhost:6379"))
```

## Quick Reference

| Feature | No State (Default) | In-Memory (`Live`) | Persistent (`Versioned`) |
|---|---|---|---|
| **Memory** | No persistence | Persists in-process | Persists across processes |
| **Object Support** | Any Python object | Any Python object | Only picklable objects |
| **Usage** | `my_task(data)` | `my_task(data, state=Live())` | `my_task(data, state=Versioned())` |
| **Checkpointing** | None | None | Automatic snapshots |
| **Best For** | Simple tasks | Multi-step workflows | Production |


## Next Steps

- **Agent Creation**: See [Agent](agent.md) for creating agents
- **Task Definition**: See [Task](task.md) for using state in agent tasks
- **Debugging**: See [View](view.md) for inspecting state
