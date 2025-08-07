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

# Get all events from state
all_events = events(state)

# Filter by specific agent using full_namespace
agent_events = [e for e in all_events if e.full_namespace == "agent_name"]
worker_events = [e for e in all_events if e.full_namespace == "orchestrator/worker"]

# Filter by namespace hierarchy
orchestrator_tree = [e for e in all_events if e.full_namespace.startswith("orchestrator")]
```

**Function Signature:**
```python
def events(state: Versioned | Live) -> list[Event]
```

**Parameters:**
- `state`: The state object to retrieve events from

**Returns:**
- `list[Event]`: All events sorted chronologically by timestamp. Use `full_namespace` field to filter by agent paths.

## Event Types

All events inherit from `BaseEvent` and include timestamps, agent name, and full namespace.

### Core Events

#### `TaskStartEvent`
Generated when an agent task begins execution.

```python
from agex.agent.events import TaskStartEvent

# Event structure
event = TaskStartEvent(
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
    parts=[...]               # list of raw output objects
)
```

#### `SuccessEvent`
Generated when a task completes successfully.

```python
from agex.agent.events import SuccessEvent

# Event structure
event = SuccessEvent(
    result="Completed!"       # Any (the actual return value of the task)
)
```

#### `FailEvent`
Generated when a task explicitly fails.

```python
from agex.agent.events import FailEvent

# Event structure
event = FailEvent(
    message="...",            # str
)
```

#### `ClarifyEvent`
Generated when a task is interrupted because the agent needs more information.

```python
from agex.agent.events import ClarifyEvent

# Event structure
event = ClarifyEvent(
    message="...",            # str
)
```

### Event Properties

All events share these common properties from `BaseEvent`:

- **`timestamp`**: `datetime` - UTC timestamp when the event occurred.
- **`agent_name`**: `str` - Name of the agent that generated the event.
- **`full_namespace`**: `str` - Complete namespace path (e.g., `"orchestrator/worker_a"`). For root-level agents, equals `agent_name`.
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

# Get all events and filter by namespace
all_events = events(state)

# Filter different levels of the hierarchy
orch_events = [
    e for e in all_events if e.full_namespace == "orchestrator"
]  # Just orchestrator

worker_events = [
    e for e in all_events if e.full_namespace.startswith("orchestrator")
]  # Orchestrator + its children

worker_a_events = [
    e for e in all_events if e.full_namespace == "orchestrator/worker_a"
]  # Just worker A
```

## Related APIs

- **[State Management](state.md)**: Understanding state containers and persistence
- **[Task Definition](task.md)**: Defining tasks and using `on_event` or `.stream()`
- **[View API](view.md)**: Experimental APIs for agent introspection

The events system forms the foundation for agent introspection and is essential for debugging, monitoring, and building sophisticated multi-agent systems.
