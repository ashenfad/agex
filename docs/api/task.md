# Task Definition (`@agent.task`)

The `@agent.task` decorator transforms function signatures into agent-driven implementations. You define the interface and behavior specification - the agent provides the implementation.

Tasks are defined on [Agent](agent.md) instances and can use [State](state.md) for persistent memory across executions.

## Basic Usage

```python
from agex import Agent

agent = Agent()

@agent.task
def solve_equation(equation: str) -> float:  # type: ignore[return-value]
    """Solve a mathematical equation and return the result."""
    pass
```

The decorated function is completely **replaced** - the agent handles all implementation.

## Decorator Patterns

### Naked Decorator
```python
@agent.task
def analyze_data(data: list[float]) -> dict:  # type: ignore[return-value]
    """Analyze numerical data and return statistics summary."""
    pass
```
Uses the function's docstring as agent instructions.

### Parameterized Decorator
```python
@agent.task("Calculate using advanced statistical methods")
def analyze_data(data: list[float]) -> dict:  # type: ignore[return-value]
    """Public API: Analyze numerical data and return statistics."""
    pass
```
- **Primer**: Instructions for the agent (first argument)
- **Docstring**: Documentation for human callers


### `setup` Parameter

The `setup` parameter runs preparatory code in the agent's sandbox *before* the agent's main execution loop begins. This is useful for providing the agent with immediate context, which can save an LLM turn.

Common use cases are to have an agent automatically inspect the head of a pandas DataFrame or view an image.

```python
from PIL.Image import Image

@agent.task(setup="view_image(inputs.image)")
def process_image(prompt: str, image: Image) -> Image:  # type: ignore[return-value]
    """Process an image based on a prompt."""
    pass

# When called, `view_image(image)` will be executed in the sandbox
# before the agent starts thinking about how to handle the prompt.
# This avoids a turn where the agent just decides to view the image.
process_image("Crop this to the subject.", image=my_image)
```

## Executing Tasks

An `@agent.task`-decorated function can be executed in two ways, depending on your needs for interactivity and observability.

### 1. Standard Execution

This is the most common way to run a task. You call the function, it blocks until the agent is finished, and then it returns the final result.

```python
result = solve_equation("2*x + 5 = 15")
print(f"Result: {result}")
```

### Async Execution

Tasks decorated on agents work seamlessly with Python's `async`/`await`. Define the task function as `async def` and `await` the result:

```python
@agent.task
async def solve_equation(equation: str) -> float:  # type: ignore[return-value]
    """Solve a mathematical equation and return the result."""
    pass

# Use with await in async context
result = await solve_equation("2*x + 5 = 15")
print(f"Result: {result}")
```

All execution modes (`on_event`, `on_token`) work with async tasks.

> [!NOTE]
> Async registered functions (via `@agent.fn`) are only available in async tasks. If an agent tries to call an async function from a sync task, it will see an error and can adapt. Use async tasks when your registered functions include async code.

### 2. Real-time Handlers with `on_event`

You may get both event-level visibility and a blocking result via the `on_event` handler. This provides a "fire-and-forget" way to get a real-time stream of all events without needing to consume a generator.

The handler is a callable that receives the raw event object each time an event is created.

```python
from agex import pprint_events

# For simple, colorful terminal logging, you can use the built-in pprint_events helper.
result = solve_equation("x**2 = 16", on_event=pprint_events)
```

## Function Signature

The decorator automatically adds `session`, `on_event`, and `on_token` parameters to your function signature as keyword-only arguments.

```python
@agent.task
def my_function(x: int, y: str) -> bool:  # type: ignore[return-value]
    """Function description."""
    pass

# Becomes callable as:
# my_function(x=10, y="hello")
# my_function(x=10, y="hello", session="user_123")
# my_function(x=10, y="hello", session="user_123", on_event=my_handler)
```

### Session Parameter

- **Optional**: `session: str = "default"`
- **Purpose**: Isolate state between different users or conversations
- **Requires**: Agent configured with `state=connect_state(...)`

```python
from agex import Agent, connect_state

agent = Agent(
    state=connect_state(type="versioned", storage="memory"),
)

@agent.task
def chat(message: str) -> str:
    """Chat with the user."""
    pass

# Different sessions have isolated memory
chat("Hello", session="user_alice")  # Alice's conversation
chat("Hello", session="user_bob")    # Bob's separate conversation

# Same session shares memory across calls
chat("Remember X=42", session="alice")
chat("What is X?", session="alice")  # Remembers X=42
```

See [State](state.md) for more details on state management.

### Concurrency Control

When using `Versioned` state with concurrent tasks (e.g., multiple workers, background jobs), tasks may conflict when trying to merge their changes. The `on_conflict` parameter controls how these conflicts are handled:

```python
# Foreground task - retry on conflict (default)
@agent.task(on_conflict='retry')
def interactive_task(query: str) -> str:  # type: ignore[return-value]
    """Process user query with automatic retry."""
    pass

# Background task - abandon work on conflict
@agent.task(on_conflict='abandon')
def background_indexing() -> None:  # type: ignore[return-value]
    """Rebuild search index in background."""
    pass

# Custom retry limit
@agent.task(on_conflict='retry', max_conflict_retries=5)
def critical_task() -> dict:  # type: ignore[return-value]
    """Important task that needs more retry attempts."""
    pass
```

**Conflict Strategies:**

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `retry` (default) | Resets state and reruns task (up to `max_conflict_retries` times) | Interactive/foreground tasks |
| `abandon` | Silently returns `None` on conflict | Background/non-critical tasks |

**How it works:**
1. Task executes and calls `snapshot()` internally after each turn
2. On completion, task attempts to `merge()` its branch to HEAD
3. If another task modified HEAD concurrently, merge fails
4. With `retry`: state is reset and task reruns from scratch
5. With `abandon`: task returns `None` and work is discarded

> [!CAUTION]
> **Side Effects and Conflicts**: Conflict handling only resets the `Versioned` state - external side effects cannot be undone. If your task calls functions that modify external systems (databases, APIs, file systems), those changes persist even when:
> - **Retry**: Task reruns from scratch (side effects from first attempt remain)
> - **Abandon**: Work is discarded (side effects already happened)
>
> This can lead to duplicate operations, orphaned data, or inconsistent state. For tasks with side effects:
> - **Prefer functional patterns**: Have agents return results that the caller uses to update external systems (after successful merge). This avoids side effects entirely.
> - Design functions to be idempotent (safe to retry/re-execute)
> - Use transactional patterns with explicit rollback
> - Consider deferring side effects until after successful merge
> - Log all external operations for manual reconciliation

See [State - Concurrent Task Handling](state.md#concurrent-task-handling) for implementation details.

### Task Cancellation

Long-running tasks can be cancelled using the `cancel()` method on the task wrapper. This gracefully stops execution at the next iteration boundary.

```python
import threading

@agent.task
def long_running_task() -> str:
    """A task that may need to be cancelled."""
    pass

# Start task in background
def run():
    try:
        result = long_running_task()
    except TaskCancelled as e:
        print(f"Cancelled after {e.iterations_completed} iterations")

thread = threading.Thread(target=run)
thread.start()

# Cancel from main thread
long_running_task.cancel()

# Or cancel a specific session
long_running_task.cancel(session="user_123")
```

**How it works:**

1. `cancel()` writes a sentinel to the underlying state store
2. The task loop checks for this sentinel at the start of each iteration
3. When detected, a `TaskCancelled` exception is raised
4. A `CancelledEvent` is recorded in the event log

**Requirements:**

- **State**: The agent must be configured with persistent state (`Versioned` with `disk` storage recommended)
- **Shared state**: For cross-thread/process cancellation, both the canceller and the running task must access the same state store

```python
from agex import Agent, connect_state, TaskCancelled

agent = Agent(
    state=connect_state(type="versioned", storage="disk", path="/tmp/agent-state"),
)
```

> [!NOTE]
> Cancellation is checked between LLM iterations, not mid-execution. If the agent is in the middle of a long function call or waiting for an LLM response, cancellation will take effect after that operation completes.


### on_event Parameter

- **Optional**: `on_event: Callable[[BaseEvent], None] | None = None`
- **Purpose**: Provide a callback function to receive events in real time.
- **Propagation**: The handler is automatically passed to any sub-agent tasks, providing a single, unified event stream for an entire end-to-end operation.

See the [Events API Guide](events.md) for more on event consumption patterns.

### on_token Parameter

- **Optional**: `on_token: Callable[[TokenChunk], None] | None = None`
- **Purpose**: Receive LLM output tokens in real time (reasoning vs. code) while still awaiting the final task result.
- **Common uses**: Live notebooks, terminal dashboards, or UI components that benefit from progressive updates.
- **Token structure**: Each callback gets a `TokenChunk` with `type`, `content`, and a `done` flag signalling the end of a section.

```python
from agex.agent import pprint_tokens

# Stream thinking/code tokens with built-in formatting
result = my_task("solve this", on_token=pprint_tokens)
```

See [Token-Level Streaming](events.md#4-token-level-streaming-with-on_token) for deeper coverage.

## Dual-Decorator Pattern

For multi-agent workflows, combine `@agent.fn` and `@agent.task` decorators:

```python
# Create specialist agents
data_processor = Agent(name="data_processor") 
orchestrator = Agent(name="orchestrator")

# Dual-decorated function: orchestrator can call data_processor's task
@orchestrator.fn(docstring="Clean and process raw data")
@data_processor.task("Remove outliers and normalize values")  
def process_data(raw_data: list[float]) -> list[float]:  # type: ignore[return-value]
    pass
```

### Decorator Order Rules
```python
# ✅ Correct order: @agent.fn OUTER, @agent.task INNER
@orchestrator.fn()
@specialist.task("Task description")
def dual_function():
    pass

# ❌ Wrong order: @agent.task before @agent.fn
@specialist.task("Task description")  
@orchestrator.fn()
def wrong_order():
    pass  # Raises ValueError
```

## Validation Rules

### Empty Function Body

Task functions must have empty bodies - the agent provides the implementation:

```python
# ✅ Valid: Empty implementations
@agent.task
def valid_function():
    """Task description."""
    pass

@agent.task  
def also_valid():
    """Another task."""
    # Comments are allowed
    pass

# ❌ Invalid: Contains implementation
@agent.task
def invalid_function():
    """This will raise an error.""" 
    return "actual code"  # Not allowed!
```

**Why empty bodies?** The decorator completely replaces your function. The agent receives your function signature and instructions, then generates code to fulfill the contract. Your implementation would be ignored anyway.

### Type Checker Compatibility

Type checkers (mypy, pylance) will complain about empty functions that promise to return values:

```python
# Type checker error: Function doesn't return anything but promises a float
@agent.task
def calculate_pi() -> float:
    """Calculate pi to high precision."""
    pass  # mypy: error - Missing return statement
```

**Solution**: Use `# type: ignore[return-value]` to silence this specific warning:

```python
@agent.task
def calculate_pi() -> float:  # type: ignore[return-value]
    """Calculate pi to high precision."""
    pass

@agent.task
def process_data(data: list[int]) -> dict:  # type: ignore[return-value]
    """Process data and return analysis."""
    pass

@agent.task  
def update_database(records: list[dict]) -> bool:  # type: ignore[return-value]
    """Update database with new records."""
    pass
```

This tells the type checker: "I know this function doesn't return what it promises, but the agent will handle it at runtime."

### Required Documentation
```python
# ✅ Valid: Has primer
@agent.task("Calculate the result")
def with_primer():
    pass

# ✅ Valid: Has docstring  
@agent.task
def with_docstring():
    """Calculate the result."""
    pass

# ❌ Invalid: No instructions
@agent.task
def no_instructions():
    pass  # Raises ValueError - no primer or docstring
```

### Single Task Decorator
```python
agent1 = Agent(name="agent1")
agent2 = Agent(name="agent2")

# Raises ValueError
@agent1.task
@agent2.task
def my_task():
    "Do a thing"
    pass
```

## Type Validation

Arguments are validated against type annotations:

```python
@agent.task
def process_numbers(data: list[int], threshold: float = 0.5) -> dict:  # type: ignore[return-value]
    """Process numerical data above threshold."""
    pass

# Validation occurs at call time
result = process_numbers([1, 2, 3], 0.8)     # ✅ Valid
result = process_numbers("invalid", 0.8)     # ❌ Raises validation error
```

## Complete Example

```python
from agex import Agent, connect_state

# Create agents with shared state configuration
state_config = connect_state(type="versioned", storage="memory")

researcher = Agent(name="researcher", state=state_config)
analyst = Agent(name="analyst", state=state_config)
coordinator = Agent(name="coordinator", state=state_config)

# Register specialist capabilities with coordinator
@coordinator.fn(docstring="Research a topic online")
@researcher.task("Search and summarize information about the given topic")
def research_topic(topic: str, depth: str = "basic") -> dict:  # type: ignore[return-value]
    """Research information about a topic."""
    pass

@coordinator.fn(docstring="Analyze research data")  
@analyst.task("Extract key insights and trends from research data")
def analyze_research(research_data: dict, focus_areas: list[str]) -> dict:  # type: ignore[return-value]
    """Analyze research findings."""
    pass

# Main coordination task
@coordinator.task("Research and analyze a topic comprehensively")
def full_research_pipeline(topic: str, focus_areas: list[str]) -> dict:  # type: ignore[return-value]
    """Complete research and analysis pipeline."""
    pass

# Execute - state is managed by the agents
result = full_research_pipeline(
    topic="renewable energy trends",
    focus_areas=["cost", "adoption", "technology"],
)

print(result)  # Comprehensive analysis from both agents
```

## Next Steps

- **[Agent](agent.md)**: Agent class and configuration
- **[Registration](registration.md)**: Expose capabilities to agents
- **[State](state.md)**: Memory, persistence, and sessions
- **[Host](host.md)**: Remote execution

