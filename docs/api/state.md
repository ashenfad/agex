# State Management

Agent state in agex enables persistent memory across [agent tasks](task.md) through a versioned, checkpointed storage system.

## Why State Management Matters

Many agent frameworks treat state management as an optional add-on or require complex configuration to achieve reliable persistence. agex provides state management as a core, first-class feature with:

- **Persistent Memory**: Agents remember variables, functions, and objects across calls
- **Automatic Checkpointing**: Every agent execution creates a commit snapshot  
- **Rollback Safety**: Debug by reverting to any previous state
- **Mutation Detection**: Automatically captures side-effect changes to prevent data loss

## `Versioned` - State Container

`Versioned` is the state container that agents use for persistent memory. Pass it to [agent tasks](task.md) via the `state` parameter:

```python
from agex import Versioned

# Create state container
state = Versioned()

# Use with agent tasks  
result = my_agent_task(data, state=state)
```

## Automatic Features

When you use `Versioned` state, you get these benefits automatically:

- **Checkpointing**: Each agent execution creates a commit snapshot
- **Mutation Detection**: Side-effect changes to objects are automatically preserved  
- **Rollback Safety**: Framework can revert to any previous state for debugging
- **Namespace Isolation**: Multi-agent workflows get separate state spaces

No additional code required - the framework handles all versioning behind the scenes.

## Storage Options

Choose the storage backend that fits your use case:

### Memory (Default)

```python
from agex import Versioned

# In-memory storage - state lost when process ends
state = Versioned()
```

**Use for:** Development, testing, ephemeral sessions

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

### Ephemeral Mode (Default)

```python
@agent.task
def analyze_data(data: list[float]) -> dict:  # type: ignore[return-value]
    """Analyze data without memory."""
    pass

# Each call is independent
result1 = analyze_data([1, 2, 3])
result2 = analyze_data([4, 5, 6])  # No memory of previous call
```

### Persistent Mode

```python
from agex import Versioned

@agent.task  
def build_analysis(data: list[float]) -> dict:  # type: ignore[return-value]
    """Build cumulative analysis with memory."""
    pass

# State persists across calls
shared_state = Versioned()
result1 = build_analysis([1, 2, 3], state=shared_state)
result2 = build_analysis([4, 5, 6], state=shared_state)  # Remembers result1
```

### Multi-Agent Coordination

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


## Next Steps

- **Agent Creation**: See [Agent](agent.md) for creating agents with state
- **Task Definition**: See [Task](task.md) for using state in agent tasks
- **Registration**: See [Registration](registration.md) for exposing capabilities to agents
- **Debugging**: See [View](view.md) for inspecting state and execution 