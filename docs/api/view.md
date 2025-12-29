# View API

!!! warning "Experimental"
    The `view()` API is an experimental debugging utility. Its output format and parameters may change in future versions.

The `view()` function provides a human-readable inspection of an agent's static capabilities or its current memory state. It is a debugging utility designed for quick, interactive use.

For programmatic access to an agent's full execution history, use the more powerful [`events()` API](events.md).

## Import

```python
from agex import view
```

## Agent Inspection: `view(agent)`

View an agent's registered capabilities to see what functions, classes, and modules are available to it. `view()` returns a formatted string.

```python
from agex import Agent, view
import math

agent = Agent(name="my_agent")

@agent.fn(visibility="high")
def calculate_sum(a: int, b: int) -> int:
    """Calculates the sum of two integers."""
    return a + b

agent.module(math, visibility="low")

# Inspect the agent's API (by default, shows high and medium visibility)
print(view(agent))

# The `full=True` flag shows all members, including low-visibility ones
# print(view(agent, full=True))

# Token budget breakdown for the static system context
print(view(agent, focus="tokens", model_name="gpt-4"))
```

**Example Output:**
```text
--- Agent API: my_agent ---
fn: calculate_sum(a: int, b: int) -> int
    Calculates the sum of two integers.
---------------------------
```

### Token Budget View

Use `focus="tokens"` to see an estimate of tokens for the agent's static system context (built-in primer, capabilities primer or rendered registrations, and agent primer). This helps you budget visibility and primer sizes before calls.

```text
--- Agent Token Budget (model: gpt-4) ---
builtin_primer: 1234
capabilities_primer: 456
agent_primer: 78
total: 1768
```

Notes:
- Counts are model-dependent; pass `model_name` to match your provider.
- Registered resources reflect current visibility (high/medium shown; low hidden).

### Notes on Recursive Module Views

If you registered a package with `recursive=True` (e.g., `osmnx` as a single module), `view(agent)`:

- Does not enumerate entire subpackages to keep the output concise and avoid heavy imports.
- Will show explicitly configured dotted members (e.g., `routing.shortest_path`) when their `visibility` is `"medium"` or `"high"`.

To ensure key nested functions appear in `view(agent)`, promote them via `configure` at registration time:

```python
import osmnx as ox
from agex import Agent, MemberSpec, view

agent = Agent()
agent.module(
    ox,
    visibility="low",
    recursive=True,
    configure={
        "geocoder.geocode": MemberSpec(visibility="high"),
        "routing.shortest_path": MemberSpec(visibility="high"),
    },
)

print(view(agent))
```

Alternatively, register submodules directly (e.g., `ox.routing`) when you want their members listed without dotted names.

## State Inspection: `view(state)` (Local Only)

View a snapshot of an agent's memory (`Versioned` or `Live` state). This is useful for debugging the outcome of an agent's execution.

Get a state object via [`agent.state()`](agent.md#statesession-str--default).

**Note:** State inspection requires local execution. For agents using remote hosts (HTTP/Modal), state is not locally accessible.

```python
from agex import Agent, connect_state, view

agent = Agent(
    state=connect_state(type="versioned", storage="memory")
)

@agent.task
def analyze_data(numbers: list[int]) -> dict:
    """Analyze data."""
    pass

# Execute tasks
analyze_data([1, 2, 3])
analyze_data([4, 5, 6])

# Inspect state (local execution only)
state = agent.state()
print(view(state))
```

**Example Output (`focus="recent"`):**
```text
--- State Diff (Commit: a1b2c3d) ---
+ ADDED: intermediate_result = [1, 4, 9]
+ ADDED: final_summary = {'sum_squares': 14}
------------------------------------
```

### State Focus Options

The `focus` parameter controls what part of the state to display:

- **`focus="recent"`** (Default): Returns a formatted **string** summarizing state changes from the most recent agent execution (like a "diff").
- **`focus="full"`**: Returns a **dictionary** containing the complete, raw key-value state at the current commit.

```python
# See what changed in the last step (returns a string)
state = agent.state()
recent_changes_view = view(state, focus="recent")
print(recent_changes_view)

# Get the entire state dictionary (returns a dict)
full_state_dict = view(state, focus="full")
if "final_summary" in full_state_dict:
    print(f"Final result: {full_state_dict['final_summary']}")
```

### Inspecting Different Sessions

Use the `session` parameter to inspect state for specific sessions:

```python
agent = Agent(state=connect_state(type="versioned", storage="memory"))

@agent.task
def chat(message: str) -> str:
    """Chat with the user."""
    pass

# Execute on different sessions
chat("Hello from Alice", session="alice")
chat("Hello from Bob", session="bob")

# Inspect each session's state
alice_state = agent.state("alice")
print("Alice's state:", view(alice_state, focus="full"))

bob_state = agent.state("bob")
print("Bob's state:", view(bob_state, focus="full"))
```

You can also use `view()` to inspect a historical state snapshot retrieved using `state.checkout()`. See [Inspecting Historical State](state.md#inspecting-historical-state) for a complete example.

## Summary of Inspection Tools

Use the right tool for the job:

| Tool | Question it Answers | Use Case |
|---|---|---|
| **`view(agent)`** | "What *can* this agent do?" | Debugging agent setup & capabilities |
| **`view(state)`** | "What is the agent's memory *right now*?" | Quick, interactive debugging of state |
| **`events(state)`** | "What did the agent *do* historically?" | Post-hoc analysis, programmatic review |

## Next Steps

- See [Agent API](agent.md) for registering capabilities
- See [State API](state.md) for state management
- See [Events API](events.md) for programmatic event access
- See [Task API](task.md) for creating agent tasks
