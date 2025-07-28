# View API

The `view()` function provides a human-readable inspection of an agent's static capabilities or its current memory state. It is a debugging utility designed for quick, interactive use.

For programmatic access to an agent's full execution history, use the more powerful [`events()` API](events.md).

## Import

```python
from agex import view
```

## Agent Inspection: `view(agent)`

View an agent's registered capabilities to see what functions, classes, and modules are available to it. This shows the "micro-DSL" you have defined for the agent.

```python
from agex import Agent, view

agent = Agent()

@agent.fn
def calculate_sum(a: int, b: int) -> int:
    return a + b

# Inspect the agent's API
api_view = view(agent)
print(api_view)

# The `full=True` flag shows all members, including low-visibility ones
complete_view = view(agent, full=True)
```

## State Inspection: `view(state)`

View a snapshot of an agent's memory (`Versioned` state). This is useful for debugging the outcome of an agent's execution.

```python
from agex import Agent, Versioned, view

agent = Agent()

@agent.task("Analyze some data")
def analyze_data(numbers: list[int], state: Versioned) -> dict:  # type: ignore[return-value]
    pass

# Initialize and execute
state = Versioned()
result = analyze_data([1, 2, 3, 4, 5], state=state)

# View a summary of the most recent state changes
state_view = view(state)
print(state_view)
```

### State Focus Options

The `focus` parameter controls what part of the state to display:

- **`focus="recent"`** (Default): Shows a summary of state changes from the most recent agent execution. This is like a "diff" of the agent's memory.
- **`focus="full"`**: Shows the complete, raw key-value state at the current commit. This is a full snapshot of the agent's memory.

```python
# See what changed in the last step
recent_changes = view(state, focus="recent")

# Get the entire state dictionary
full_state = view(state, focus="full")
if "result" in full_state:
    print(f"Final result: {full_state['result']}")
```

You can also use `view()` to inspect a historical state snapshot retrieved using `state.checkout()`. See [Inspecting Historical State](state.md#inspecting-historical-state) for a complete example.

## Summary of Inspection Tools

Use the right tool for the job:

| Tool | Question it Answers | Use Case |
|---|---|---|
| **`view(agent)`** | "What *can* this agent do?" | Debugging agent setup & capabilities |
| **`view(state)`** | "What is the agent's memory *right now*?" | Quick, interactive debugging of state |
| **`events(state)`** | "What did the agent *do* historically?" | Post-hoc analysis, programmatic review |
| **`task.stream()`** | "What is the agent *doing* right now?" | Real-time, interactive event streaming |

## Next Steps

- See [Agent API](agent.md) for registering capabilities
- See [State API](state.md) for state management
- See [Events API](events.md) for programmatic event access
- See [Task API](task.md) for creating agent tasks 