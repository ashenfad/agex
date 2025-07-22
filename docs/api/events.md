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
    state: Versioned | Ephemeral | Namespaced, 
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

All events inherit from `BaseEvent` and include automatic timestamps and agent attribution:

### Core Events

#### `TaskStartEvent`
Generated when an agent task begins execution.

```python
from agex.agent.events import TaskStartEvent

# Event structure
event = TaskStartEvent(
    agent_name="my_agent",
    task_name="process_data", 
    inputs={"data": "value"},
    message="Process the input data and return results."
)
```

#### `ActionEvent` 
Generated when an agent takes an action (thinks + executes code).

```python
from agex.agent.events import ActionEvent

# Event structure  
event = ActionEvent(
    agent_name="my_agent",
    thinking="I need to calculate the sum of these numbers.",
    code="result = sum([1, 2, 3, 4, 5])\nprint(f'Sum: {result}')"
)
```

#### `OutputEvent`
Generated when agents produce output (print, view_image, etc.).

```python
from agex.agent.events import OutputEvent

# Event structure
event = OutputEvent(
    agent_name="my_agent",
    parts=[PrintAction(["Hello, world!"])]  # Raw objects stored
)
```

#### `SuccessEvent` 
Generated when a task completes successfully.

```python
from agex.agent.events import SuccessEvent

# Event structure
event = SuccessEvent(
    agent_name="my_agent",
    result="Processing completed successfully!"
)
```

#### `FailEvent`
Generated when a task explicitly fails.

```python
from agex.agent.events import FailEvent

# Event structure
event = FailEvent(
    agent_name="my_agent", 
    message="Could not process input: invalid format"
)
```

### Event Properties

All events share these common properties from `BaseEvent`:

- **`timestamp`**: UTC timestamp (datetime) when the event occurred
- **`agent_name`**: Name of the agent that generated the event
- **`commit_hash`**: If using `Versioned` state, the commit hash of the agent's state before this event occurred. See [Inspecting Historical State](../api/state.md#inspecting-historical-state) for how to use this for debugging.

## Consuming Events

There are three primary ways to consume events from agent tasks, depending on whether you need real-time access or post-hoc analysis.

### 1. Post-Hoc Analysis with `events()`

The `events()` function is the ideal tool for analyzing a task after it has completed. You pass it the `state` object used during the run, and it returns a complete, chronologically sorted list of all events that occurred, including those from sub-agents.

```python
from agex import events, Versioned

state = Versioned()
result = my_task("run analysis", state=state)

# After the task is done, get all events for analysis
all_events = events(state)
action_events = [e for e in all_events if isinstance(e, ActionEvent)]
print(f"The agent took {len(action_events)} actions.")
```
This is the primary method for debugging and detailed inspection of an agent's behavior.

### 2. Real-time Streaming with `.stream()`

For interactive use cases, like in a Jupyter notebook, you can use the `.stream()` method on a task. It returns a generator that yields events as they happen.

See the **[Streaming Execution Guide in the Task API Docs](task.md#2-streaming-execution-with-stream)** for full details.

### 3. Real-time Handlers with `on_event`

For production monitoring and integration with observability platforms, you can pass an `on_event` handler to a task. This provides a "fire-and-forget" callback that receives every event in real time for the entire execution, including all sub-agent events.

See the **[Real-time Handlers Guide in the Task API Docs](task.md#3-real-time-handlers-with-on_event)** for full details.

## Usage Patterns

### Basic Event Retrieval

```python
from agex import Agent, Versioned, events

# Create agent and state
agent = Agent(name="math_agent")
state = Versioned()

@agent.task
def calculate(x: int, y: int) -> int:
    """Calculate x + y."""
    pass

# Execute task
result = calculate(x=5, y=3, state=state)

# Get all events
all_events = events(state)
print(f"Generated {len(all_events)} events")

# Get events from specific agent only (no children)
agent_events = events(state, "math_agent", children=False)
print(f"Math agent generated {len(agent_events)} events")
```

### Event Type Filtering

```python
from agex.agent.events import ActionEvent, OutputEvent, SuccessEvent

# Get all events
all_events = events(state, "math_agent", children=False)

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
from agex import Agent, Versioned, events

# Create orchestrator and worker agents
orchestrator = Agent(name="orchestrator")
worker_a = Agent(name="worker_a") 
worker_b = Agent(name="worker_b")

state = Versioned()

@orchestrator.task
def coordinate_work(task_data: dict) -> dict:
    """Coordinate work between multiple agents."""
    pass

@worker_a.task  
def process_part_a(data) -> str:
    """Process part A of the work."""
    pass

@worker_b.task
def process_part_b(data) -> str:
    """Process part B of the work."""
    pass

# Execute multi-agent workflow
result = coordinate_work({"data": "sample"}, state=state)

# Monitor different levels
all_events = events(state)                                    # Everything
orch_events = events(state, "orchestrator", children=False)   # Just orchestrator
worker_events = events(state, "orchestrator")                 # Orchestrator + workers
worker_a_events = events(state, "orchestrator", "worker_a", children=False)  # Just worker A
```

## Error Handling

Events are always safely retrievable, even when agents encounter errors:

```python
from agex import Agent, TaskFail, events

agent = Agent(name="error_prone")

@agent.task  
def might_fail() -> str:
    """Task that might fail."""
    pass

try:
    result = might_fail(state=state)
except TaskFail:
    # Even failed tasks generate events
    failure_events = events(state, "error_prone", children=False)
    print(f"Failure analysis: {len(failure_events)} events generated")
```

## Related APIs

- **[State Management](state.md)**: Understanding state containers and persistence
- **[Agent](agent.md)**: Creating agents that generate events  
- **[Task](task.md)**: Defining tasks that create TaskStartEvent and completion events
- **[View](view.md)**: Experimental APIs for agent introspection

The events system forms the foundation for agent introspection and is essential for debugging, monitoring, and building sophisticated multi-agent systems. 