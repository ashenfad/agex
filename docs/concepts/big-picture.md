# The Big Picture: Agents That Think in Code

Many agentic frameworks adopt JSON tooling as the communication medium between your code and AI agents. This choice often means designing new tool-friendly abstractions and serialization when working with existing codebases. 

`agex` takes a different approach: **agents work directly with your Python runtime**. Instead of JSON interfaces, agents think in code and operate on real Python objects.

## Core Philosophy: Code as the Language of Reasoning

The key insight is that **code is language made formal enough to get stuff done**. The same tools that help human developers manage complexity work naturally for AI agents:

- **REPLs** for interactive exploration and step-by-step reasoning
- **`dir()` and `help()`** for discovering capabilities  
- **State inspection** like `print` for understanding structure
 - **Imports for whitelisted modules** for accessing functionality
- **Function definitions** for building reusable tools

Instead of inventing new agent interaction patterns, `agex` adapts the proven tools developers have used for decades. This makes agents more effective and their behavior more predictable.

## Implementation Foundation

This philosophy shapes how agex works:

- **REPL-Like Environment** - Agents operate in familiar, persistent environments where they can introspect (`help`, `dir`), see their own output (`print`, `view_image`), and build solutions iteratively
- **Natural Error Handling** - Validation errors and exceptions appear in the agent's stdout just like in a real Python environment, creating natural debugging loops
- **Flexible State Management** - Tasks can run in live mode or with persistent state that captures the agent's entire workspace, enabling both simple single-shot tasks and complex workflows where agents keep and re-use their own functions


## The Middle Road: Guidance Through a Curated Environment

Agentic frameworks often present a stark choice: provide agents with rigid, pre-defined tools, or grant them access to a full, open-ended compute environment. The former offers guidance at the cost of flexibility, while the latter provides power at the cost of reliability and focus.

`agex` is designed to be the middle road.

The whitelist registration system is more than just a security feature; it is a tool for **guidance**. By carefully selecting which functions, classes, and modules you expose, you are effectively designing a **"micro-DSL" (Domain-Specific Language)** for your agent.

This curated environment helps guide the agent toward a correct solution by limiting its scope of action to only the most relevant capabilities. It prevents the agent from getting lost in the vastness of a full compute environment and encourages it to compose the building blocks you provide. This "micro-DSL" can be as small or as large as you need, from a handful of functions to broad access to a library, giving you fine-grained control over the balance of guidance and freedom.

This philosophy of providing guidance through a curated environment is the primary design principle. A powerful and welcome side-effect of this approach is a robust security model. By limiting the agent's world to only the capabilities you provide, you inherently prevent it from accessing unintended, and potentially unsafe, parts of your system. Security becomes a natural outcome of thoughtful agent design.

### Registration, Not Tool-Making

This curated environment is created through **registration**, not by writing tool abstractions. In many frameworks, adapting a library for agent use requires writing wrapper functions or "tools" that handle JSON serialization and provide a simplified interface. `agex` bypasses this entirely.

Instead of writing `tools/my_pandas_tool.py`, you simply register the `pandas` library directly with the agent:

```python
import pandas as pd
agent.module(pd)  # see agex/helpers/pandas_helper.py for a full example
```

Your existing codebase *is* the toolset. The registration system acts as a secure and targeted bridge between your code and the agent, without forcing you to create and maintain a parallel set of tool abstractions.

For a more detailed comparison of this approach to industry-standard tooling models like MCP, see our full **[note on this philosophy](agex-and-mcp.md)**.

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

Git-like versioning with automatic checkpointing enables powerful debugging:

```python
from agex import ActionEvent, Versioned, events

state = Versioned()
result = my_agent_task("complex analysis", state=state)

# Every agent action creates a versioned checkpoint
all_events = events(state)
action_event = next(e for e in all_events if isinstance(e, ActionEvent))

# Time-travel to see exactly what the agent was thinking
historical_state = state.checkout(action_event.commit_hash)
```

This creates a complete audit trail where you can inspect the agent's workspace at any point in its reasoning process.

### 4. Multi-Agent Orchestration

Natural coordination through hierarchical delegation or simple Python control flow. No workflow DSLs or configuration files needed - just Python.

**Hierarchical Delegation**

The dual-decorator pattern (`@orchestrator.fn` + `@specialist.task`) enables elegant specialist architectures where sub-agents become callable functions for an orchestrator. This allows for natural hierarchies where complex workflows feel like simple function composition.

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

## The Result

agex transforms agent development from framework-specific tooling into natural Python programming. Multi-agent workflows become simple control flow. Complex data handoffs become object passing. Agent capabilities become library registrations.

The result is a more natural division of labor: developers provide curated access to powerful libraries, and agents take on the work of composing those libraries into novel solutions.
