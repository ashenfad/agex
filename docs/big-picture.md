# The Big Picture: Agents That Think in Code

This document explains the vision and architectural principles behind `agex`. Unlike frameworks that constrain agents to JSON tool calls, `agex` gives agents a familiar Python environment where they can think, explore, and build solutions using real code.

## Core Philosophy: Code as the Language of Reasoning

The key insight is that **code is language made formal enough to get stuff done**. The same tools that help human developers manage complexity work naturally for AI agents:

- **REPLs** for interactive exploration and step-by-step reasoning
- **`dir()` and `help()`** for discovering capabilities  
- **State inspection** for understanding what's available
- **Modular imports** for accessing functionality
- **Function definitions** for building reusable tools

Instead of inventing new agent interaction patterns, `agex` adapts the proven tools developers have used for decades. This makes agents more effective and their behavior more predictable.

## Implementation Foundation

### REPL-Like Agent Environment

Each agent operates in a familiar, persistent environment where they can introspect (`help`, `dir`), see recent state changes, view their own output, and build solutions iteratively. This cognitive scaffolding mirrors how human developers work and makes agent behavior more predictable.

### Smart Context Management

The framework automatically manages token budgets and context windows - recent information gets priority, older context gets compressed, and agents maintain continuity without manual intervention. This eliminates the typical context management burden while keeping agents focused on relevant information.

### Natural Error Handling

Validation errors and exceptions appear in the agent's stdout just like in a real Python environment. Agents can see what went wrong and retry within bounded limits, creating a natural debugging loop that feels familiar to developers.

### Flexible State Management

Tasks can run in ephemeral mode (no memory) or with persistent state across calls. Multiple agents can collaborate through isolated but shareable state, enabling both simple single-shot tasks and complex multi-agent workflows.

## Runtime Interoperability

### Seamless Python Integration

A key differentiator of this framework is **runtime interoperability** - agents don't just execute code in isolation, they create objects that live and work directly in your Python runtime.

**True Callable Generation:**
```python
from agex import Agent
from typing import Callable

my_coder = Agent(name="coder")

@my_coder.task
def make_a_function(prompt: str) -> Callable:  # type: ignore[return-value]
    """Generate a Python function from a text description."""
    pass

# Returns an actual Python callable you can use immediately
prime_finder = make_a_function("Find the next prime larger than a given number")
next_prime = prime_finder(100)  # Works with existing code
my_list.sort(key=prime_finder)  # Integrates with standard library
```

### Beyond Tool Isolation

Most agent frameworks force a choice between limited string/JSON processing or complete VM isolation. This framework provides a third option: **runtime integration** where agents create real Python objects that participate directly in your existing codebase.

**Data Processing Handoffs:**
```python
import pandas as pd
from agex import Agent

# Seamless data flow between your context and agent context
messy_dataframe = pd.read_csv("complex_data.csv")

data_agent = Agent(name="data_processor")

@data_agent.task
def clean_and_analyze(df: pd.DataFrame) -> dict:  # type: ignore[return-value]
    """Clean a pandas DataFrame and extract analytical insights."""
    pass

insights = clean_and_analyze(messy_dataframe)
# insights is a real dict in your session - no serialization needed
```

**Dynamic Code Extension:**
```python
# Agent extends your existing classes with new capabilities
@my_coder.task
def add_method_to_class(cls: type, method_description: str) -> type:  # type: ignore[return-value]
    """Dynamically add a new method to an existing class."""
    pass

EnhancedProcessor = add_method_to_class(MyDataProcessor, "add outlier detection")
# Your class now has the new method, usable immediately
```

### Natural Agent Orchestration

Because agents return real Python objects, complex multi-agent workflows become simple Python control flow:

```python
# Generator-critique loop using standard Python
# (assuming agent tasks are already defined with @agent.task)
rpt = research_expert("please research ...")
while not (judgement := judge(rpt)).approved:
    rpt = research_revise(judgement.feedback)

# Parallel processing with list comprehensions  
analyses = [specialist_agent(data_chunk) for data_chunk in dataset]

# Conditional branching based on agent outputs
if classifier_agent(document).confidence > 0.8:
    result = expert_agent(document)
else:
    result = human_review_agent(document)
```

No workflow DSLs or configuration files needed - just Python.

### Living Codebase Integration

This enables workflows impossible with isolated execution: agents become collaborative development partners who extend existing systems at runtime, evolve code without breaking interfaces, and orchestrate through natural Python control flow. The result is agents that don't just help *with* your code - they become part of your development environment.

## Hierarchical Agent Architecture

### Agent-to-Agent Communication

Functions can be decorated as both capabilities and tasks:

```python
from agex import Agent

orchestrator = Agent(name="orchestrator")
research_expert = Agent(name="research_expert")

@orchestrator.fn(docstring="Conduct comprehensive research on the given topic")
@research_expert.task("Perform deep research with academic rigor")
def deep_research(topic: str) -> ResearchReport:  # type: ignore[return-value]
    """Conduct comprehensive research on the given topic."""
    pass
```

This enables natural hierarchies where specialist agents serve as capabilities for generalist orchestrators.

### Side-Channel Communication

**(Future Feature)** Agents could communicate through a `log()` builtin:

```python
# Speculative future feature
log("Found 12 relevant papers, starting analysis", to="parent_agent")
log("Need human guidance on conflicting sources", to="system")
log("Focusing on theory, suggest you handle applications", to="analysis_agent")
```

Messages would appear in the target agent's stdout, fitting naturally into the REPL environment. Role-based targeting would provide security boundaries and clear communication channels.

### Shared but Namespaced State

Multiple agents can collaborate while maintaining isolation:
- Shared underlying state with agent-specific namespaces
- Cross-agent communication through explicit channels (function calls, logging)
- Clean separation prevents accidental interference
- Enables complex multi-agent workflows with clear boundaries



## Long-Term Agent Evolution (Future Roadmap)

### Adaptive Memory Management

As agents accumulate extensive interaction histories, the framework could implement **exponentially decaying memory** - recent conversations get full fidelity, older interactions get progressively compressed, and ancient history becomes high-level summaries. This would allow agents to maintain both detailed recent context and long-term learning without overwhelming token budgets.

### Self-Refactoring Agents

Long-running agents could be given explicit tasks to **review and refactor their own accumulated code**:

```python
@agent.task("Review and refactor your helper functions for better organization.")
def refactor_my_toolkit() -> None:  # type: ignore[return-value]
    """Reorganize accumulated tools, abstract patterns, improve naming."""
    pass
```

Over time, agents would develop personal coding styles and specialized toolkits, becoming more effective not just at tasks but at self-improvement.

### Agent-Designed Agent Architecture

The most speculative possibility: agents creating their own specialist sub-agents on demand, analyzing their task patterns to architect cognitive division of labor, and designing hierarchical specialist networks. This could transform agents from "users of tools" into "architects of intelligence" - responsible for both solving problems and designing the cognitive structures to solve them effectively.



## Security & Boundaries

### Natural Security Points

The registration system provides clean security boundaries:
- Explicit capability registration prevents unauthorized access
- Role-based communication limits cross-agent interference  
- Namespaced state prevents accidental data leakage
- Validation ensures type safety at agent boundaries

### Trust Boundaries

Agents can only call explicitly registered functions, creating clear trust boundaries. The visibility system provides additional control over what capabilities are exposed to which agents.

 