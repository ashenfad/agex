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


## Executing Tasks

An `@agent.task`-decorated function can be executed in three ways, depending on your needs for interactivity and observability.

### 1. Standard Execution

This is the most common way to run a task. You call the function, it blocks until the agent is finished, and then it returns the final result.

```python
result = solve_equation("2*x + 5 = 15")
print(f"Result: {result}")
```

### 2. Streaming Execution with `.stream()`

For interactive scenarios like Jupyter notebooks, you can use the `.stream()` method. This returns a **generator** that yields events as they happen, allowing you to see the agent's progress in real time.

The `.stream()` method accepts the exact same arguments as the original task function.

```python
# In a Jupyter notebook
from IPython.display import display

for event in solve_equation.stream("4x + 2 = 10"):
    display(event) # Renders a rich view of each event
```

The final result of the task is not returned directly, but is available as the `.result` attribute of the `SuccessEvent` yielded at the end of the stream.

### 3. Real-time Handlers with `on_event`

You may get both event-level visibility and a blocking result via the `on_event` handler. This provides a "fire-and-forget" way to get a real-time stream of all events without needing to consume a generator.

The handler is a callable that receives the raw event object each time an event is created.

```python
from agex import pprint_events

# For simple, colorful terminal logging, you can use the built-in pprint_events helper.
result = solve_equation("x**2 = 16", on_event=pprint_events)
```

## Function Signature

The decorator automatically adds `state` and `on_event` parameters to your function signature as keyword-only arguments.

```python
@agent.task
def my_function(x: int, y: str) -> bool:  # type: ignore[return-value]
    """Function description."""
    pass

# Becomes callable as:
# my_function(x=10, y="hello")
# my_function(x=10, y="hello", state=my_state)
# my_function(x=10, y="hello", state=my_state, on_event=my_handler)
```

### State Parameter

- **Optional**: `state: Versioned | Live | None = None`
- **One-shot mode** (default): No memory between calls
- **Persistent mode**: Pass a `Versioned` or `Live` state for long-term memory

```python
from agex import Versioned

# Persistent state across multiple calls  
shared_state = Versioned()
result1 = my_function(x=10, y="hello", state=shared_state)
result2 = my_function(x=20, y="world", state=shared_state)  # Remembers previous call
```

See [State](state.md) for more details on state management.

### on_event Parameter

- **Optional**: `on_event: Callable[[BaseEvent], None] | None = None`
- **Purpose**: Provide a callback function to receive events in real time.
- **Propagation**: The handler is automatically passed to any sub-agent tasks, providing a single, unified event stream for an entire end-to-end operation.

See the [Events API Guide](events.md) for more on event consumption patterns.

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
from agex import Agent, Versioned

# Create agents
researcher = Agent(name="researcher")
analyst = Agent(name="analyst") 
coordinator = Agent(name="coordinator")

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

# Execute with persistent state
shared_state = Versioned()
result = full_research_pipeline(
    topic="renewable energy trends",
    focus_areas=["cost", "adoption", "technology"],
    state=shared_state
)

print(result)  # Comprehensive analysis from both agents
```

## Next Steps

- **Agent Creation**: See [Agent](agent.md) for Agent class documentation
- **Registration**: See [Registration](registration.md) for exposing capabilities to agents
- **State Management**: See [State](state.md) for `Versioned` objects and persistent agent memory
- **Debugging**: See [View](view.md) for inspecting task execution and state changes
