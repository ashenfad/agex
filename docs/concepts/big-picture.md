# The Big Picture: Code as the Only Interface

Many agentic frameworks use JSON tooling as the communication layer between your code and AI agents. Agents select from predefined tools, pass JSON-serialized arguments, and receive JSON-serialized results. This creates a boundary: your Python objects must be converted to JSON and back, and you must write tool definitions that wrap your existing code.

`agex` eliminates this boundary entirely: **agents operate exclusively in Python**. There are no tools to define, no JSON to serialize, no selection step. Agents write Python code that runs directly in your runtime and works with real Python objects.

## Core Philosophy: Python is the Agent's Native Language

Agents don't choose between "using tools" and "writing code." In agex, **everything is code**:

- Returning a result: `task_success("hello")`
- Calling a function: just call it
- Building data structures: native Python syntax
- Exploring capabilities: `dir()` and `help()`
- Debugging: `print()` to see output
- Creating reusable logic: define functions

This isn't code generation as a feature—it's code as the fundamental medium. Agents operate in a sandboxed Python environment where the same patterns developers use (introspection, iteration, composition) work naturally for AI.

## How It Works

Agents operate in a **generate → execute → observe** loop:

1. **Generate**: The LLM generates a block of Python code based on the task and available capabilities
2. **Execute**: The framework runs this code in a secure sandbox with only whitelisted capabilities
3. **Observe**: Output (prints, errors, return values) flows back to the agent for the next iteration

This creates a natural development experience:

- **Iterative refinement** - Agents see their output, adjust their approach, and try again
- **Natural error handling** - Exceptions appear in stdout just like in a real Python environment
- **Persistent workspace** - With versioned state, agents can define functions, store variables, and build solutions across multiple iterations
- **Familiar debugging** - `print()`, `dir()`, `help()` work as expected


## The Middle Road: Guidance Through a Curated Environment

Agentic frameworks often present a stark choice: provide agents with rigid, pre-defined tools, or grant them access to a full, open-ended compute environment. The former offers guidance at the cost of flexibility, while the latter provides power at the cost of reliability and focus.

`agex` is designed to be the middle road.

The whitelist registration system is more than just a security feature; it is a tool for **guidance**. By carefully selecting which functions, classes, and modules you expose, you are providing conceptual guide-rails for your agent.

This curated environment helps lead the agent toward a correct solution by limiting its scope of action to only the most relevant capabilities. It prevents the agent from getting lost in the vastness of a full compute environment and encourages it to compose the building blocks you provide.

### Registration, Not Tool-Making

This curated environment is created through **registration**, not by writing tool abstractions. In many frameworks, adapting a library for agent use requires writing wrapper functions or "tools" that handle JSON serialization and provide a simplified interface. `agex` bypasses this entirely.

Instead of writing `tools/my_pandas_tool.py`, you simply register the `pandas` library directly with the agent:

```python
import pandas as pd
agent.module(pd)
```

Your existing codebase *is* the toolset. The registration system acts as a secure and targeted bridge between your code and the agent, without forcing you to create and maintain a parallel set of tool abstractions.

#### Leveraging the Existing Ecosystem

The premise is simple: a vast ecosystem of Python libraries already exists.

Rather than wrapping logic in new abstractions, `agex` lets agents use existing libraries directly—the same ones you'd use in regular Python code.

## Security Through Design

The registration system provides clean security boundaries:

- **Explicit capability registration** - Agents can only access functions you explicitly expose
- **Visibility controls** - Fine-grained control over what capabilities are prominent vs. hidden
- **Namespace isolation** - Multiple agents work with shared state without interference
- **Type validation** - Automatic validation ensures data integrity at agent boundaries

## Tangibly Different

This overall philosophy manifests in ways that distinguish agex from traditional frameworks:

### 1. Runtime Interoperability

Agents create real Python objects that live in your runtime, not isolated JSON responses:

```python
import math
from typing import Callable
from agex import Agent

agent = Agent()
agent.module(math)

@agent.task
def make_function(description: str) -> Callable:  # type: ignore[return-value]
    """Generate a Python function from a text description."""
    pass

# Agent uses math module to build the requested function
distance_fn = make_function(
    "takes a 2D point (x, y) and returns distance from the origin"
)

# Returns an actual Python callable you can use immediately
points = [(4, 3), (1, 1), (0, 5)] 
points.sort(key=distance_fn)  # Works with existing Python code
```

No JSON serialization overhead. No wrapper classes. Just real Python objects flowing between your code and agents.

### 2. Code-as-Action

Instead of rigid pre-built tools, agents compose primitives into solutions:

```python
import statistics
from agex import Agent

agent = Agent()
agent.module(statistics)  # Give building blocks

@agent.task
def analyze(data: list[float], request: str) -> dict[str, float]:  # type: ignore[return-value]
    """Analyze data to fulfill the user's request."""
    pass

# Agent composes statistics.mean() and statistics.median() 
# in a single execution to handle complex requests
result = analyze(
    [1, 2, 3, 4, 5, 6, 100], 
    "What are the mean and median for only positive numbers?"
)
```

Traditional frameworks would likely require separate tool calls for filtering and various aggregations (mean and median). agex agents compose operations efficiently within single executions.

### 3. Agent Workspace Persistence  

Git-like versioning with automatic checkpointing enables detailed debugging:

```python
from agex import Agent, ActionEvent, connect_state, events

agent = Agent(
    state=connect_state(type="versioned", storage="memory"),
)

# ... define and run a task ...
result = my_agent_task("complex analysis")

# Every agent action creates a versioned checkpoint
state = agent.state()
all_events = events(state)
action_event = next(e for e in all_events if isinstance(e, ActionEvent))

# Time-travel to see exactly what the agent was thinking
historical_state = state.checkout(action_event.commit_hash)
```

This creates a complete audit trail where you can inspect the agent's workspace at any point in its reasoning process.

### 4. Multi-Agent Orchestration

Natural coordination through hierarchical delegation or simple Python control flow. No workflow DSLs or configuration files needed - just Python.

**Hierarchical Delegation**

The dual-decorator pattern (`@orchestrator.fn` + `@specialist.task`) enables specialist architectures where sub-agents become callable functions for an orchestrator. This allows for natural hierarchies where complex workflows feel like simple function composition.

```python
# Sub-agents as functions for orchestrators
@orchestrator.fn
@specialist.task
def process_data(data: list) -> dict:  # type: ignore[return-value]
    "Clean and normalize data"
    pass
```

**Peer Collaboration**

Agents can also collaborate as peers using standard Python control flow, such as in an iterative improvement loop:

```python
# Iterative improvement between agents  
report = research("AI trends in 2024")
while not (review := critique(report)).approved:
    report = hone_report(review.feedback, report)
```

### 5. Agent-Authored Libraries

Agents aren't limited to using the tools you provide—they can build their own. They can create persistent modules in their Virtual Filesystem (VFS) and `import` them in subsequent iterations. 

This enables agents to manage complexity by:
- **Organizing code** into reusable helper modules
- **Building domain-specific abstractions** on the fly
- **Persisting logic** across multiple turns without re-generating it

A "Workspace Recap" automatically provides the agent with an inventory of its self-authored modules, including function signatures and class definitions, ensuring it always knows what tools it has built for itself.

## The Result

With agex, multi-agent workflows become simple control flow. Complex data handoffs become object passing. Agent capabilities become library registrations.

The result is a straightforward division of labor: developers provide curated access to libraries, and agents compose those libraries into solutions.
