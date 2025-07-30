# Events System

The events system in agex provides comprehensive introspection into agent execution, capturing everything from task starts to outputs and errors. Events enable debugging, monitoring, streaming, and building sophisticated multi-agent coordination.

## Overview

Every agent action generates **typed events** that are stored in the agent's state and can be retrieved for analysis, debugging, or real-time monitoring. The unified `events()` API provides flexible access to events across namespace hierarchies.

**Key Benefits:**
- **Complete Visibility**: See exactly what agents are thinking and doing
- **Real-time Monitoring**: Stream events as agents execute
- **Debugging**: Analyze agent behavior and decision patterns
- **Multi-agent Coordination**: Monitor complex agent interactions
- **Chronological Ordering**: Events are automatically timestamped and sorted

## Core API

### `events()` - Unified Event Access

The `events()` function is the primary interface for retrieving events from agent state:

```python
from agex import events

# Basic usage
all_events = events(state)                          # Current namespace + children (default)
direct_events = events(state, children=False)      # Current namespace only

# Namespace navigation  
agent_events = events(state, "agent_name")         # Navigate to agent + children
worker_events = events(state, "orch", "worker")    # Navigate to nested path + children
isolated_events = events(state, "agent", children=False)  # Just that agent, no sub-agents
```

**Function Signature:**
```python
def events(
    state: Versioned | Live | Namespaced, 
    *namespaces: str, 
    children: bool = True
) -> list[Event]
```

**Parameters:**
- `state`: The state object to retrieve events from
- `*namespaces`: Variable number of namespace path components to navigate to
- `children`: Whether to include events from child namespaces (default: `True`)

**Returns:**
- `list[Event]`: Events sorted chronologically by timestamp

## Event Types

All events inherit from `BaseEvent` and include automatic timestamps and agent attribution.

### Core Events

#### `TaskStartEvent`
Generated when an agent task begins execution.

```python
from agex.agent.events import TaskStartEvent

# Event structure
event = TaskStartEvent(
    agent_name="my_agent",    # str
    task_name="process_data", # str
    inputs={"data": "value"}, # dict[str, Any]
    message="...",            # str
)
```

#### `ActionEvent`
Generated when an agent takes an action (thinks + executes code).

```python
from agex.agent.events import ActionEvent

# Event structure
event = ActionEvent(
    agent_name="my_agent",    # str
    thinking="...",           # str
    code="..."                # str
)
```

#### `OutputEvent`
Generated when agents produce output (print, view_image, etc.).

```python
from agex.agent.events import OutputEvent

# Event structure
event = OutputEvent(
    agent_name="my_agent",    # str
    parts=[...]               # list of raw output objects
)
```

#### `SuccessEvent`
Generated when a task completes successfully.

```python
from agex.agent.events import SuccessEvent

# Event structure
event = SuccessEvent(
    agent_name="my_agent",    # str
    result="Completed!"       # Any (the actual return value of the task)
)
```

#### `FailEvent`
Generated when a task explicitly fails.

```python
from agex.agent.events import FailEvent

# Event structure
event = FailEvent(
    agent_name="my_agent",    # str
    message="...",            # str
)
```

#### `ClarifyEvent`
Generated when a task is interrupted because the agent needs more information.

```python
from agex.agent.events import ClarifyEvent

# Event structure
event = ClarifyEvent(
    agent_name="my_agent",    # str
    message="...",            # str
)
```

### Event Properties

All events share these common properties from `BaseEvent`:

- **`timestamp`**: `datetime` - UTC timestamp when the event occurred.
- **`agent_name`**: `str` - Name of the agent that generated the event.
- **`commit_hash`**: `str | None` - If using `Versioned` state, the commit hash of the agent's state before this event occurred. See [Inspecting Historical State](state.md#inspecting-historical-state) for how to use this for debugging.

## Consuming Events

There are three primary ways to consume events from agent tasks.

### 1. Post-Hoc Analysis with `events()`
This is the ideal tool for analyzing a task **after it has completed**. You pass the `state` object used during the run, and it returns a complete, chronologically sorted list of all events that occurred, including those from sub-agents.

This is the primary method for debugging and detailed inspection of an agent's behavior.

```python
from agex import events, Versioned
from agex.agent.events import ActionEvent

state = Versioned()
result = my_task("run analysis", state=state)

# After the task is done, get all events for analysis
all_events = events(state)
action_events = [e for e in all_events if isinstance(e, ActionEvent)]
print(f"The agent took {len(action_events)} actions.")
```

### 2. Real-time Callback with `on_event`
The `on_event` parameter is the recommended approach for most real-time use cases. It provides a true, real-time stream of events as they happen—even from sub-agents—while preserving the natural flow of a standard function call.

**Choose `on_event` if:**
*   You need the final return value of the task.
*   You want a simple "fire-and-forget" callback for logging or display.
*   You require immediate, non-batched events from hierarchical agent workflows.

**In Jupyter notebooks:**
```python
from IPython.display import display

# See events pop up in real-time while getting the final result
result = my_task("analyze this data", on_event=display)
print(f"Final result: {result}")
```

**For production monitoring:**
```python
from agex.agent.events import FailEvent

def custom_handler(event):
    # Custom processing logic for production monitoring
    if isinstance(event, FailEvent):
        send_alert(event.message)
    log_to_observability_platform(event)

result = my_task("important task", on_event=custom_handler)
```

### 3. Real-time Generator with `.stream()`
The `.stream()` method is best for situations where your primary goal is to process the stream of events itself, and the final return value of the task is **not** needed.

**Choose `.stream()` if:**
*   You only care about the events and not the final result.
*   You want to use generator-based control flow (e.g., with `itertools`).

```python
from agex.agent.events import ActionEvent

# You get a generator of events, but not the final result.
for event in my_task.stream("process data"):
    if isinstance(event, ActionEvent):
        log_agent_action(event)
```

**Known Limitation:** In hierarchical agent workflows, events from a sub-agent are currently yielded as a **single batch** after the sub-agent task completes, rather than being streamed one-by-one. For true real-time streaming in multi-agent setups, prefer `on_event`.

## Usage Patterns

### Event Type Filtering

```python
from agex.agent.events import ActionEvent, OutputEvent, SuccessEvent

# Get all events
all_events = events(state)

# Filter by event type
action_events = [e for e in all_events if isinstance(e, ActionEvent)]
output_events = [e for e in all_events if isinstance(e, OutputEvent)]
success_events = [e for e in all_events if isinstance(e, SuccessEvent)]

print(f"Actions: {len(action_events)}")
print(f"Outputs: {len(output_events)}")  
print(f"Successes: {len(success_events)}")
```

### Multi-Agent Event Monitoring

```python
# Assume orchestrator, worker_a, and worker_b agents are defined
# and a multi-agent task has been run with a `state` object.

# Monitor different levels of the hierarchy
all_events = events(state)                                    # Everything
orch_events = events(state, "orchestrator", children=False)   # Just orchestrator
worker_events = events(state, "orchestrator")                 # Orchestrator + its children
worker_a_events = events(state, "orchestrator", "worker_a", children=False)  # Just worker A
```

## Related APIs

- **[State Management](state.md)**: Understanding state containers and persistence
- **[Task Definition](task.md)**: Defining tasks and using `on_event` or `.stream()`
- **[View API](view.md)**: Experimental APIs for agent introspection

The events system forms the foundation for agent introspection and is essential for debugging, monitoring, and building sophisticated multi-agent systems.
