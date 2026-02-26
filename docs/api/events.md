# Events System

The events system in agex provides comprehensive introspection into agent execution, capturing everything from task starts to outputs and errors. Events enable debugging, monitoring, streaming, and building sophisticated multi-agent coordination.

## Overview

Every agent action generates **typed events** that are stored in the agent's state and can be retrieved for analysis, debugging, or real-time monitoring. The `events()` API provides access to events from an agent's state.

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
```

**Function Signature:**
```python
def events(state: Staged | Live) -> list[Event]
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
Generated when an agent takes an action (thinks + executes code or terminal commands).

```python
from agex.agent.events import ActionEvent

# Event structure
event = ActionEvent(
    title="...",             # str
    thinking="...",          # str
    code="...",              # str | None - Python code to execute
    terminal="...",          # str | None - Terminal commands to execute
    file_actions=[...],      # list[FileAction | EditAction]
    input_tokens=...,        # int | None - Actual input tokens from LLM API
    output_tokens=...,       # int | None - Actual output tokens from LLM API
)
```

**Action Types:**
- **Python execution**: `code` contains Python code, `terminal` is `None`
- **Terminal execution**: `terminal` contains shell commands, `code` is `None`

Terminal commands are executed against the agent's virtual filesystem and support common Unix utilities like `ls`, `cat`, `grep`, `find`, `tar`, `gzip`, `zip`, etc. Terminal execution implicitly continues the task loop (no `task_success()` needed).

```python
# Check what type of action was taken
if event.code:
    print(f"Python: {event.code}")
elif event.terminal:
    print(f"Terminal: {event.terminal}")
```

#### `FileAction`
Used within `ActionEvent` to represent a file write or append requested by the agent.

```python
from agex.agent.datatypes import FileAction

# Structure
action = FileAction(
    path="utils.py",         # str
    content="...",           # str
    mode="write"             # Literal["write", "append"]
)
```

#### `EditAction`
Used within `ActionEvent` to represent a search/replace or insert edit requested by the agent.

```python
from agex.agent.datatypes import EditAction

# Structure
action = EditAction(
    path="utils.py",         # str
    search="old_code",       # str - text to find
    content="new_code",      # str - content to use
    operation="replace",     # Literal["replace", "insert-after", "insert-before"]
    match_all=False,         # bool - apply to all occurrences
)
```

**Operation Options:**
- `"replace"` (default): The search text is replaced entirely by the content
- `"insert-after"`: The search text is kept, content is inserted after it
- `"insert-before"`: Content is inserted before the search text, which is kept

The `operation` attribute controls how the edit is applied. Use `match_all=True` to apply the operation to all occurrences of the search text.

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

#### `SummaryEvent`
Generated when the event log exceeds the `log_high_water_tokens` threshold (see [Agent - Event Log Summarization](agent.md#event-log-summarization)). The LLM condenses older events into a summary, preserving essential context while reducing tokens.

```python
from agex.agent.events import SummaryEvent

# Event structure
event = SummaryEvent(
    summary="...",                  # str - The condensed summary text
    summarized_event_count=10,      # int - Number of events summarized
    original_tokens=5000,           # int - Token cost of original events
    low_detail_threshold=datetime(...),  # datetime | None - Events older than this render at low detail
)
```

When a `SummaryEvent` is present, agex applies 3-tier rendering: **Summarized** (oldest, replaced by compact text), **Low Detail** (compressed images/nesting), **Full Detail** (newest, complete rendering).

#### `FileEvent`
Generated when files are added, modified, or removed from the workspace.

```python
from agex.agent.events import FileEvent

# Event structure
event = FileEvent(
    file_source="agent",     # Literal["user", "agent"]
    added=["new.py"],        # list[str]
    modified=["utils.py"],   # list[str]
    removed=["temp.txt"],    # list[str]
)
```

**File Source:**
- `"agent"`: Emitted as a **single aggregated event** at task completion.
- `"user"`: Emitted **per operation** via `agent.fs()` calls.

#### `ErrorEvent`
Generated for framework-level errors (e.g., LLM API failures) that agents don't handle directly. Used by the retry system to log each failed attempt.

```python
from agex.agent.events import ErrorEvent

# Event structure
event = ErrorEvent(
    error=...,               # Any - The actual exception object
    recoverable=True,        # bool - Whether the task can continue
)
```

When LLM retries are enabled, each failed attempt emits an `ErrorEvent` with `recoverable=True`. If all retries are exhausted, a final `ErrorEvent` with `recoverable=False` is emitted before raising `LLMFail`. See [Error Handling - LLMFail](errors.md#llmfail-framework) for details.

#### `CancelledEvent`
Generated when a task is cancelled via an external request (e.g., `my_task.cancel()`).

```python
from agex.agent.events import CancelledEvent

# Event structure
event = CancelledEvent(
    task_name="...",              # str - Name of the cancelled task
    iterations_completed=3,      # int - How many iterations completed before cancellation
)
```

See [Task - Task Cancellation](task.md#task-cancellation) for details on the cancellation mechanism.

### Event Properties

All events share these common properties from `BaseEvent`:

- **`timestamp`**: `datetime` - UTC timestamp when the event occurred.
- **`agent_name`**: `str` - Name of the agent that generated the event.
- **`full_namespace`**: `str` - The agent's namespace path. Equals `agent_name` for the agent that owns the state.
- **`commit_hash`**: `str | None` - The commit hash linking this event to `Versioned` state. Only populated when using `Versioned` state (see [State Management](state.md)); `None` for `Live` or ephemeral state.
  - **For action events** (`ActionEvent`, `OutputEvent`, etc.): The commit hash *before* the action—useful for inspecting what the agent saw when it made a decision.
  - **For task result events** (`SuccessEvent`, `FailEvent`, `ClarifyEvent`, `CancelledEvent`): The commit hash *after* the result is recorded—enabling "reset to this outcome" workflows via `state.reset_to(event.commit_hash)`.
- **`source`**: `Literal["setup", "main"]` - The execution phase that generated the event. Defaults to `"main"`. Events generated by the `setup` parameter of `@agent.task` are tagged with `"setup"`.
- **`full_detail_tokens`**: `int` - Cached token estimate for full-detail rendering. Computed automatically at event creation.
- **`low_detail_tokens`**: `int` - Cached token estimate for low-detail rendering (used when event age triggers compression). Typically 25-50% of `full_detail_tokens`. Computed automatically for `TaskStartEvent`, `OutputEvent`, and `SuccessEvent`; equals `full_detail_tokens` for other event types.

## Consuming Events

There are three primary ways to consume events from agent tasks.

### 1. Post-Hoc Analysis with `events()`
This is the ideal tool for analyzing a task **after it has completed**. You pass the `state` object used during the run, and it returns a complete, chronologically sorted list of all events that occurred, including those from sub-agents.

This is the primary method for debugging and detailed inspection of an agent's behavior.

```python
from agex import events, Staged
from agex.agent.events import ActionEvent

state = Staged()
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

### 3. Token-Level Streaming with `on_token`
`on_token` is an optional callback that receives LLM output tokens in real time. Tokens arrive as lightweight `TokenChunk` objects with:

- `type`: one of `"title"`, `"thinking"`, `"file"`, `"edit"`, `"terminal"`, or `"python"`
- `content`: the text fragment for that section
- `done`: a boolean that signals the end of the current section
- `input_tokens`: actual input token count from the LLM API (set on the final chunk only)
- `output_tokens`: actual output token count from the LLM API (set on the final chunk only)

**Token Types:**
| Type | Description |
|------|-------------|
| `"title"` | Optional title for the action |
| `"thinking"` | Agent's reasoning/planning text |
| `"file"` | File creation content (`<FILE>` tag) |
| `"edit"` | File edit content (`<EDIT>` tag) |
| `"terminal"` | Shell commands to execute (`<TERMINAL>` tag) |
| `"python"` | Python code to execute (`<PYTHON>` tag) |

> **Note:** `"terminal"` and `"python"` are mutually exclusive—the agent uses one or the other per turn, never both.

**Choose `on_token` if:**
*   You want progressive UI feedback while the LLM is generating content.
*   You need to distinguish between the agent's reasoning and emitted code.
*   You are building terminal dashboards or notebooks that benefit from sub-second updates.

```python
from agex.agent import pprint_tokens

# Stream thinking/code tokens with colorized terminal output
result = my_task("generate code", on_token=pprint_tokens)

# Custom handler for a UI
from agex.llm.core import TokenChunk

def render_token(chunk: TokenChunk):
    if chunk.type == "thinking":
        ui.update_thinking(chunk.content)
    elif chunk.type in ("python", "terminal"):
        ui.update_code(chunk.content)
    if chunk.done:
        ui.section_complete(chunk.type)

result = my_task("analyze", on_token=render_token)
```

Token streaming activates only when an `on_token` handler is provided. When omitted, tasks behave exactly as before.

### 4. Async Handlers (sync handlers also work)

All event consumption patterns work with async tasks. For async-first codebases:

```python
@agent.task
async def my_async_task(data: str) -> str:  # type: ignore[return-value]
    """An async task."""
    pass

# Async with callbacks
result = await my_async_task("process this", on_event=my_handler)
```

The `on_event` and `on_token` callbacks can also be async functions—agex will `await` them automatically.

### 5. Console Pretty-Printing Helpers

Use the top-level helpers to get colorful terminal output without writing custom handlers.

```python
from agex import pprint_events
from agex.agent import pprint_tokens

# Real-time: pass as on_event/on_token
result = my_task(
    "analyze",
    on_event=pprint_events,
    on_token=pprint_tokens,
)

# Post-hoc: pretty-print all events from state
from agex import events
all_events = events(state)
pprint_events(all_events, verbosity="brief")
```

`pprint_events` works with a single event, any iterable/generator of events, or as an `on_event` handler. `pprint_tokens` focuses on streaming tokens and prints the full content of each chunk.

Notes:
- Both helpers respect the `color="auto" | "always" | "never"` setting and the `NO_COLOR` environment variable.
- `pprint_events` keeps a running Δ time between prints when used as `on_event`.
- `pprint_tokens` renders content with colors: 💭 thinking in blue, 🐍 Python in yellow, 💻 terminal in green.
- `pprint_tokens` ignores section-complete markers (`done=True`) so streams remain tidy.

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

### Filtering Setup Events

Events generated during the `setup` phase (e.g., initial context loading) are tagged with `source="setup"`. You can filter these out to focus on the agent's main execution loop.

```python
# Get all events
all_events = events(state)

# Filter out setup events (e.g., for a clean UI display)
main_events = [e for e in all_events if e.source != "setup"]

# Or specifically check for setup output
setup_outputs = [
    e for e in all_events 
    if e.source == "setup" and isinstance(e, OutputEvent)
]
```

### Multi-Agent Event Monitoring

In multi-agent workflows, each agent stores events in its own state:

```python
# Get events from each agent via agent.state()
orchestrator_events = events(orchestrator.state())
worker_events = events(worker.state())
```

## Related APIs

- **[State Management](state.md)**: Understanding state containers and persistence
- **[Task Definition](task.md)**: Defining tasks and using `on_event`
- **[View API](view.md)**: Experimental APIs for agent introspection

The events system forms the foundation for agent introspection and is essential for debugging, monitoring, and building sophisticated multi-agent systems.
