# State Management

Agent state in agex gives you flexible control over memory and persistence across [agent tasks](task.md). You can choose between live execution (default) for maximum flexibility, or persistent state for complex multi-step workflows.

## State Options Overview

agex provides three approaches to state management:

### **1. Live (Default) - No State Parameter**
- **No persistence**: Variables don't survive between task calls
- **Maximum flexibility**: Agents can work with any Python object, including unpicklable ones
- **Simplicity**: No state management needed - just call your tasks
- **Use for**: Simple, one-off tasks, prototyping, working with complex objects

### **2. Live with Memory - `Live()` Object**
- **Process-bound memory**: Agents remember variables across calls within the same process
- **Maximum flexibility**: Agents can work with any Python object, including unpicklable ones
- **No checkpointing**: No rollback capabilities or automatic snapshots
- **Process-only**: Memory lost when process ends
- **Use for**: Multi-step workflows that need memory but don't require persistence or rollback

### **3. Persistent State - `Versioned()` Object**
- **Cross-process persistence**: Agents remember variables across calls and process restarts
- **Automatic checkpointing**: Every agent execution creates a commit snapshot  
- **Rollback safety**: Debug by reverting to any previous state
- **Mutation detection**: Automatically captures side-effect changes to prevent data loss
- **Constraints**: Objects must be picklable for persistence
- **Use for**: Production workflows, multi-step processes, agent coordination, debugging

## State Containers

### `Live` - Process-Bound Memory

`Live` provides memory within the current process without persistence or checkpointing:

```python
from agex import Live

# Create live state container
state = Live()

# Use with agent tasks  
result = my_agent_task(data, state=state)
```

### `Versioned` - Persistent Memory

`Versioned` is the state container that agents use for persistent memory with checkpointing:

```python
from agex import Versioned

# Create persistent state container
state = Versioned()

# Use with agent tasks  
result = my_agent_task(data, state=state)
```

## Persistent State Features

When you explicitly use `Versioned` state, you get these benefits automatically:

- **Checkpointing**: Each agent execution creates a commit snapshot
- **Mutation Detection**: Side-effect changes to objects are automatically preserved  
- **Rollback Safety**: Framework can revert to any previous state for debugging
- **Namespace Isolation**: Multi-agent workflows get separate state spaces

No additional code required - the framework handles all versioning behind the scenes when you pass a `Versioned` state object.

## Inspecting Historical State

A key feature of `Versioned` state is the ability to inspect the agent's workspace at a prior point in time. Because every agent execution creates a commit, the agent's history can be navigated much like a git repository.

Every event in the [Events API](./events.md) is stamped with the `commit_hash` of the state as it existed just before that event occurred. This enables a debugging workflow for "time-travel" inspection:

1.  **Inspect the event log** to find an interesting agent action.
2.  **Get the `commit_hash`** from that event.
3.  **Checkout the historical state** to see the exact memory the agent had when it made its decision.

```python
from agex import events, view

# 1. Find an event of interest after a run
all_events = events(state)
action_event = next(e for e in all_events if isinstance(e, ActionEvent))

# 2. Get the commit hash from that event
commit_to_inspect = action_event.commit_hash

# 3. Checkout the state to a new variable
historical_state = state.checkout(commit_to_inspect)
# `historical_state` is a read-only view of the past.
# The original `state` object is unchanged.
print(f"--- Inspecting state at {commit_to_inspect[:7]} ---")
print(view(historical_state, focus="full"))
```

This allows for detailed inspection of an agent's historical state to understand its decision-making process.

## Storage Options

Choose the storage backend that fits your use case:

### Memory (Default)

```python
from agex import Versioned

# In-memory storage - state lost when process ends
state = Versioned()
```

**Use for:** Development, testing, live sessions

### Disk Storage

```python
from agex import Versioned, Disk

# Persistent storage - state survives process restarts
state = Versioned(Disk("/path/to/storage"))

# Optional size limit (default: 1GB)
state = Versioned(Disk("/path/to/storage", size_limit=500*1024*1024))
```

**Use for:** Production deployments, long-running workflows

### Cached Disk Storage

```python
from agex import Versioned, Cache, Disk

# High-performance: memory cache + disk persistence
disk_store = Disk("/path/to/storage")
state = Versioned(Cache(disk_store, max_bytes=64*1024*1024))
```

**Use for:** Performance-critical applications with large state

## Using State with Agent Tasks

### 1. Live (Default) - No State Parameter

When you don't pass a `state` parameter, each call is completely independent:

```python
@agent.task
def analyze_data(data: list[float]) -> dict:  # type: ignore[return-value]
    """Analyze data without memory."""
    pass

# Each call is independent - no state management needed
result1 = analyze_data([1, 2, 3])
result2 = analyze_data([4, 5, 6])  # No memory of previous call

# ✅ Agents can work with any Python object, including unpicklable ones
@agent.task
def process_db_data(query: str) -> dict:  # type: ignore[return-value]
    """Process database query results."""
    pass

# Works with database cursors, file handles, iterators, etc.
result = process_db_data("SELECT * FROM users")
```

### 2. Live with Memory - `Live()` Object

When you pass an `Live()` state object, agents remember variables across calls within the same process:

```python
from agex import Live

@agent.task  
def build_analysis(data: list[float]) -> dict:  # type: ignore[return-value]
    """Build cumulative analysis with memory."""
    pass

# State persists across calls within the same process
shared_state = Live()
result1 = build_analysis([1, 2, 3], state=shared_state)
result2 = build_analysis([4, 5, 6], state=shared_state)  # Remembers result1

# ✅ Agents can work with any Python object, including unpicklable ones
@agent.task
def process_cursors(query: str) -> dict:  # type: ignore[return-value]
    """Process and remember database cursors."""
    pass

# Can store database cursors, file handles, etc. in variables
result = process_cursors("SELECT * FROM users", state=shared_state)
```

### 3. Persistent State - `Versioned()` Object

When you pass a `Versioned()` state object, agents get persistent memory with automatic checkpointing:

```python
from agex import Versioned

@agent.task  
def build_analysis(data: list[float]) -> dict:  # type: ignore[return-value]
    """Build cumulative analysis with memory."""
    pass

# State persists across calls and process restarts
shared_state = Versioned()
result1 = build_analysis([1, 2, 3], state=shared_state)
result2 = build_analysis([4, 5, 6], state=shared_state)  # Remembers result1

# ⚠️ Objects must be picklable for persistence
# Database cursors, file handles, etc. cannot be assigned to variables
# but can still be used via direct method chaining
```

### Choosing Between Approaches

**Use live (no state) when:**
- Building simple, one-off tasks
- Prototyping or experimenting
- You don't need memory between task calls

**Use live with memory when:**
- Building multi-step workflows that need memory
- Working with complex objects (database connections, file handles)
- You don't need persistence across process restarts
- You don't need rollback/debugging capabilities

**Use persistent state when:**
- Building production workflows
- You need persistence across process restarts
- Coordinating multiple agents
- You need rollback/debugging capabilities

### Multi-Agent Coordination

Multi-agent workflows require persistent state for memory sharing and coordination:

```python
from agex import Agent, Versioned

# Create specialist agents
data_processor = Agent(name="data_processor")
coordinator = Agent(name="coordinator")

# Dual-decorated function for inter-agent communication
@coordinator.fn(docstring="Process raw data")
@data_processor.task("Clean and normalize data")
def process_data(data: list) -> dict:  # type: ignore[return-value]
    pass

@coordinator.task("Run data pipeline")
def pipeline(raw_data: list) -> dict:  # type: ignore[return-value]
    pass

# Agents share state with automatic namespace isolation
# Note: Multi-agent coordination requires persistent state
shared_state = Versioned()
result = pipeline([1, 2, 3], state=shared_state)
```

## Custom Storage Backends

For distributed or specialized storage needs, implement the `KVStore` interface:

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

| Feature | Live (Default) | Live with Memory | Persistent State |
|---------|---------------------|----------------------|------------------|
| **Memory** | No persistence between calls | Variables persist within process | Variables persist across processes |
| **Object Support** | Any Python object | Any Python object | Only picklable objects |
| **Usage** | `my_task(data)` | `my_task(data, state=Live())` | `my_task(data, state=Versioned())` |
| **Checkpointing** | None | None | Automatic snapshots |
| **Rollback** | Not available | Not available | Full rollback support |
| **Process Restart** | N/A | Memory lost | Memory preserved |
| **Multi-Agent** | Not supported | Basic sharing | Automatic namespace isolation |
| **Best For** | Simple tasks | Multi-step workflows, complex objects | Production workflows, coordination |


## Next Steps

- **Agent Creation**: See [Agent](agent.md) for creating agents with state
- **Task Definition**: See [Task](task.md) for using state in agent tasks
- **Registration**: See [Registration](registration.md) for exposing capabilities to agents
- **Debugging**: See [View](view.md) for inspecting state and execution 