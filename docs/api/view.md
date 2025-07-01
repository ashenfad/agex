# View API

⚠️ **Experimental API**: The `view()` function is the least developed interface in agex and will likely change frequently as the framework evolves.

The `view()` function provides human-readable inspection of agents and their execution state.

## Import

```python
from agex import view
```

## Agent Inspection

View an agent's registered capabilities:

```python
from agex import Agent, view

agent = Agent()

@agent.fn
def calculate_sum(a: int, b: int) -> int:
    return a + b

# Inspect the agent's API
api_view = view(agent)
print(api_view)

# Full view shows all members including low-visibility
complete_view = view(agent, full=True)
```

**Parameters:**
- `obj: Agent` - The agent to inspect
- `full: bool = False` - Show all members regardless of visibility

## State Inspection

View agent execution state:

```python
from agex import Agent, Versioned, view

agent = Agent()

@agent.task("Analyze some data")
def analyze_data(numbers: list[int], state: Versioned) -> dict:
    pass  # type: ignore[return-value]

# Initialize and execute
state = Versioned()
result = analyze_data([1, 2, 3, 4, 5], state=state)

# View state
state_view = view(state)
print(state_view)
```

**Parameters:**
- `obj: Versioned` - The state object to inspect
- `focus: "recent" | "full" | "stdout" = "recent"` - What to show
- `model_name: str = "gpt-4"` - Tokenizer for "recent" view
- `max_tokens: int = 4096` - Token budget for "recent" view

### Focus Options

**`focus="recent"`** (default): Recent state changes as formatted summary
**`focus="full"`**: Complete state as dictionary 
**`focus="stdout"`**: Agent print output as list of strings

```python
recent_view = view(state, focus="recent")      # What changed
full_state = view(state, focus="full")         # All variables
stdout_log = view(state, focus="stdout")       # Print output
```

## Common Usage

```python
# Debug agent capabilities
print(view(agent))

# Check what agent did
print(view(state, focus="recent"))

# Access specific state values
full_state = view(state, focus="full")
if "result" in full_state:
    print(f"Result: {full_state['result']}")

# See agent's print statements
for output in view(state, focus="stdout"):
    print(f"Agent: {output}")
```

## Next Steps

- See [Agent API](agent.md) for registering capabilities
- See [State API](state.md) for state management
- See [Task API](task.md) for creating agent tasks 