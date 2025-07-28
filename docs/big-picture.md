# The Big Picture: Agents That Think in Code

Many agentic frameworks adopt JSON tooling as the communication medium to the LLM. This
choice means designing tool-friendly hi-level abstractions when sharing pre-existing codebases with agents. `agex` takes a different approach: **agents work directly with your Python runtime**. Agents can consume fns and are themselves functions (the tasks they are assigned).

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

The framework automatically manages token budgets and context windows - recent information gets priority, older context gets compressed (**(Future Feature)**), and agents maintain continuity without manual intervention. This eliminates the typical context management burden while keeping agents focused on relevant information.

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

### Multi-modal Reasoning

Because agents execute code, they can tap into the full multi-modal capabilities of the underlying LLM. Rather than being limited to text or JSON, agents can directly "see" and reason about images generated within their environment.

This is enabled by a built-in `view_image()` function. When an agent generates a plot, chart, or any other image, it can pass that image to `view_image()` to have it included in the context for its next reasoning step.

**Example: Self-Correcting Visualization**

```python
# Agent generates a plot using a library like matplotlib
plt.figure()
plt.plot(data)
plt.title("Initial Sales Data")

# Agent "views" the plot to analyze it
view_image(plt.gcf())

# Based on what it sees, the agent might think:
# "The title is not descriptive enough. I will add a better title and a y-axis label."

# Agent then generates corrected code:
plt.title("Quarterly Sales Performance (2023)")
plt.ylabel("Revenue (USD)")
view_image(plt.gcf()) # Views the improved plot to confirm
```

This creates a powerful feedback loop where an agent can generate visual information, critique its own work, and iteratively improve it—all within a single task. This capability emerges naturally from the code-centric design of `agex`, avoiding the need for specialized "vision tools" and allowing agents to work with information in a more human-like way.

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

### Storage, State, and Concurrency

A core design principle in `agex` is that state should be both secure and flexible. This is achieved through a deliberate architectural choice: the **serialization boundary**. When an agent completes an evaluation cycle (a single step in its reasoning loop), any data stored in a persistent `Versioned` object is serialized.

This approach has several advantages:

- **Security & Rollback**: Serialization isolates the agent's execution from the host's runtime, preventing unwanted side effects. It also creates atomic, versioned snapshots of the state, allowing for easy rollbacks if a task fails or produces undesirable results.
- **Future-Proofing for Distribution**: This boundary is what makes future distributed execution models (e.g., running agents in separate processes or on different machines) possible.
- **Expanded Data Capacity**: While there is some overhead to serialization, it allows agents to work with complex Python objects (`numpy` arrays, `pandas` DataFrames, custom classes) that are orders of magnitude larger and more complex than what is feasible with standard JSON-based approaches.

This state-snapshotting model also provides a simple and powerful guarantee for multi-agent systems: **atomicity**. In hierarchical agent architectures, state is only saved after the top-level agent task completes. This effectively creates a transaction, preventing race conditions where multiple sub-agents might otherwise attempt to write to the same state concurrently.

This design simplifies reasoning about state in complex orchestrations. However, it also means that `agex` is currently best suited for hierarchical or sequential agent workflows, as opposed to "swarm" scenarios where multiple sibling agents need to collaborate on a shared state asynchronously. This versioning is robust enough to handle distributed workers for the same agent. If two independent processes were to modify the same state, the `Versioned` store behaves like a distributed version control system (e.g., Git). Rather than corrupting data, each process would create a distinct, parallel history—effectively creating branches. This guarantees data integrity, though it means the application logic may need to reconcile these different "heads" later, similar to a `git merge` operation.

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

**Update**: A minimal implementation of this concept now exists in [`examples/dogfood.py`](../examples/dogfood.py), where agents can create other agents at runtime using natural language prompts.

The broader possibility remains speculative: agents creating their own specialist sub-agents on demand, analyzing their task patterns to architect cognitive division of labor, and designing hierarchical specialist networks. This could transform agents from "users of tools" into "architects of intelligence" - responsible for both solving problems and designing the cognitive structures to solve them effectively.

While the core mechanism works, the utility of recursive agent creation in practical applications is still unproven but intriguing.

## Security & Boundaries

### Natural Security Points

The registration system provides clean security boundaries:
- Explicit capability registration prevents unauthorized access
- Role-based communication limits cross-agent interference  
- Namespaced state prevents accidental data leakage
- Validation ensures type safety at agent boundaries

### Trust Boundaries

Agents can only call explicitly registered functions, creating clear trust boundaries. The visibility system provides additional control over what capabilities are exposed to which agents.

 