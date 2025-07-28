# The Big Picture: Agents That Think in Code

Many agentic frameworks adopt JSON tooling as the communication medium between your code and AI agents. This choice forces you to design tool-friendly abstractions and handle complex serialization when working with existing codebases. 

`agex` takes a different approach: **agents work directly with your Python runtime**. Instead of JSON interfaces, agents think in code and operate on real Python objects.

## Core Philosophy: Code as the Language of Reasoning

The key insight is that **code is language made formal enough to get stuff done**. The same tools that help human developers manage complexity work naturally for AI agents:

- **REPLs** for interactive exploration and step-by-step reasoning
- **`dir()` and `help()`** for discovering capabilities  
- **State inspection** for understanding what's available
- **Modular imports** for accessing functionality
- **Function definitions** for building reusable tools

Instead of inventing new agent interaction patterns, `agex` adapts the proven tools developers have used for decades. This makes agents more effective and their behavior more predictable.

## Implementation Foundation

This philosophy shapes how agex works:

- **REPL-Like Environment** - Agents operate in familiar, persistent environments where they can introspect (`help`, `dir`), see their own output (`print`, `view_image`), and build solutions iteratively
- **Natural Error Handling** - Validation errors and exceptions appear in the agent's stdout just like in a real Python environment, creating natural debugging loops
- **Smart Context Management** - The framework automatically manages token budgets, keeping agents focused on relevant information without manual intervention
- **Flexible State Management** - Tasks can run in live mode or with persistent state that captures the agent's entire workspace, enabling both simple single-shot tasks and complex multi-agent workflows where agents keep and re-use their own functions

## The agex Advantage

This philosophy manifests in four core capabilities that distinguish agex from traditional frameworks:

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

No serialization overhead. No wrapper classes. Just real Python objects flowing between your code and agents.

### 2. Code-as-Action

Instead of rigid pre-built tools, agents compose primitives into solutions:

```python
import statistics
from agex import Agent

agent = Agent()
agent.module(statistics)  # Give building blocks

@agent.task
def analyze(data: list[float], request: str) -> dict:  # type: ignore[return-value]
    """Analyze data to fulfill the user's request."""
    pass

# Agent composes statistics.mean() and statistics.median() 
# in a single execution to handle complex requests
result = analyze([1, 2, 3, 4, 5, 6, 100], 
                "What are the mean and median for only positive numbers?")
```

Traditional frameworks require separate tool calls for mean and median. agex agents compose operations efficiently within single executions.

### 3. Agent Workspace Persistence  

Git-like versioning with automatic checkpointing enables powerful debugging:

```python
from agex import Versioned, events

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

Natural coordination through hierarchical delegation or simple Python control flow:

```python
# Hierarchical: Sub-agents as functions for orchestrators
@orchestrator.fn(docstring="Process raw data") 
@specialist.task("Clean and normalize data")
def process_data(data: list) -> dict:  # type: ignore[return-value]
    pass

# Python control flow: Iterative improvement between agents  
report = researcher("AI trends in 2024")
while not (review := critic(report)).approved:
    report = writer(review.feedback, report)
```

No workflow DSLs or configuration files needed - just Python.

## Why This Matters

### Beyond Tool Isolation

Most frameworks force a choice between limited JSON processing or complete VM isolation. agex provides a third option: **runtime integration** where agents participate directly in your existing codebase while maintaining security through controlled capability exposure.

### Natural Multi-Modal Reasoning

Because agents execute code, they can work with any Python library that generates visual content:

```python
# Agent generates a visualization
plt.figure()
plt.plot(data)
plt.title("Sales Analysis")

# Agent "sees" its own work and iterates
view_image(plt.gcf())
# Agent might think: "Title needs improvement"
```

This creates natural feedback loops where agents can critique and improve their own visual outputs within a single task.

### Hierarchical Agent Architecture

The dual-decorator pattern enables elegant specialist architectures:

```python
orchestrator = Agent(name="orchestrator")
research_expert = Agent(name="research_expert")

@orchestrator.fn(docstring="Conduct comprehensive research")
@research_expert.task("Perform deep research with academic rigor")
def deep_research(topic: str) -> ResearchReport:  # type: ignore[return-value]
    """Research a topic thoroughly."""
    pass
```

Specialist agents become callable functions for orchestrator agents, enabling natural hierarchies where complex workflows feel like simple function composition.

## Security Through Design

The registration system provides clean security boundaries:

- **Explicit capability registration** - Agents can only access functions you explicitly expose
- **Visibility controls** - Fine-grained control over what capabilities are prominent vs. hidden
- **Namespace isolation** - Multiple agents work with shared state without interference
- **Type validation** - Automatic validation ensures data integrity at agent boundaries

## The Result

agex transforms agent development from framework-specific tooling into natural Python programming. Multi-agent workflows become simple control flow. Complex data handoffs become object passing. Agent capabilities become library registrations.

The framework disappears into the background, leaving you with agents that think and work like experienced Python developers.
