# The Big Picture

Most agent frameworks ask you to define tools — JSON schemas wrapping your code, with the agent picking from a list and arguments serializing back and forth across the boundary on each call. `agex` doesn't have that boundary. You define a typed Python function with `@task`, and the agent fills it in by writing sandboxed Python that calls into the modules you've registered. Real Python objects (DataFrames, Plotly figures, your Pydantic models) flow back into your code unchanged. Your existing codebase *is* the toolset.

Three structural choices follow from that — a typed-function contract, a library shape, and a pure-Python sandbox that runs anywhere Python runs — and they're what distinguishes agex from other agent frameworks.

## Code as the medium

Agents don't choose between "using tools" and "writing code." In agex, everything is code:

- Returning a result: `task_success(...)`
- Calling a function: just call it
- Building data structures: native Python syntax
- Debugging: `print()` and read the output next turn
- Creating reusable logic: define helpers and drop them in `helpers/`

Agents operate in a **generate → execute → observe** loop:

1. **Generate** — the LLM writes a block of Python based on the task and registered capabilities.
2. **Execute** — the framework runs the block in a sandbox.
3. **Observe** — output (prints, errors, return values) flows back as the next turn's context.

Errors land in stdout the way they would in a normal Python session — the agent sees the traceback, adjusts, tries again. No special "error-handling tool" needed.

## Three pillars

### 1. Typed function as the contract

You declare what the task does with a function signature; the agent fills in the body.

```python
import pandas as pd
from agex import Agent

agent = Agent()
agent.module(pd)

@agent.task
def summarize(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Compute summary statistics for the named columns."""
    pass

stats = summarize(my_df, ["price", "revenue"])  # real dict[str, float]
```

The return type is part of the contract. agex validates the agent's `task_success(...)` value against the annotation; if it doesn't match, the agent sees a type error and tries again. Rich types — DataFrames, Plotly figures, Pydantic models, callables — flow back into your code as actual Python objects, with no JSON intermediary.

This is the load-bearing primitive. The action space (sandboxed Python), the state model (filesystem + cache + event log), and the orchestration story (sub-agents as functions) all follow from it.

### 2. Library, not service

agex is a Python library. You import it, register your existing modules, and define `@task` functions. The agent runs inside your application's Python process.

```python
import agex
from your_project import analytics

agent = agex.Agent()
agent.module(analytics)

@agent.task
def report(question: str) -> str:
    """Answer a question using the analytics module."""
    pass
```

There's no separate runtime to deploy, no API endpoint to call, no IPC boundary between your code and the agent's. When the agent returns a `pd.DataFrame`, it's the same object the next line of your code can pass to `df.to_csv()`.

This is the opposite shape from standalone-agent frameworks (smolagents, Claude Code, etc.), which run as their own processes and communicate via text or files. agex is closer in shape to a typed function library — the agent is something you *call*, not something you converse with.

### 3. Pure-Python sandbox, runs anywhere

The sandbox is implemented as AST rewriting in pure Python ([sandtrap](https://github.com/ashenfad/sandtrap)). The default in-process mode runs entirely inside your Python interpreter — no subprocess, no kernel syscalls, no external runtime. That structural choice has a downstream consequence: the same agent code runs in-process, in a subprocess, in a kernel-isolated worker (seccomp/Landlock/Seatbelt), or — via [Pyodide](https://pyodide.org/) — entirely in a browser tab.

```python
agent = Agent(
    isolation="none",      # in-process (default)
    # isolation="process",   # subprocess
    # isolation="kernel",    # subprocess + seccomp / Landlock / Seatbelt
)
```

[agex-studio](https://agex.studio) is the proof-of-concept: a chat-driven data analysis app with pandas, scikit-learn, plotly, calendar integration, and an interactive preview pane — all client-side, no backend. It's the same agex you'd embed in a server-side application; the only difference is the runtime.

The agent's environment is identical across modes. The cache, VFS, and event log work the same way; under process or kernel isolation, host-side resources are reached via an internal RPC channel so the agent code doesn't have to change.

## Registration: guidance and security

Agent frameworks usually present a stark choice — rigid pre-defined tools, or a fully open compute environment. agex is the middle road.

The whitelist registration system serves two roles. **Guidance**: by selecting which functions, classes, and modules you expose, you give the agent guide-rails that steer it toward correct solutions. **Security**: agents can only access what you explicitly register, with fine-grained visibility controls and type validation at boundaries.

```python
import pandas as pd
agent.module(pd)
```

Instead of writing a tool wrapper around pandas, you register pandas directly. Visibility settings let you bury low-level helpers and surface high-value APIs.

## What this enables

Several capabilities fall out naturally from the three pillars.

**Multi-agent orchestration with regular Python control flow.** Sub-agents are decorated functions; orchestrators call them like any other. No workflow DSL, no graph builder.

```python
@orchestrator.fn
@specialist.task
def process_data(data: list) -> dict:
    """Clean and normalize data."""
    pass
```

Peer collaboration uses ordinary Python loops:

```python
report = research("AI trends in 2025")
while not (review := critique(report)).approved:
    report = hone_report(review.feedback, report)
```

**Agent-authored libraries.** Agents can write helper modules to the Virtual Filesystem (`helpers/utils.py`) and `import` them in subsequent tasks — useful for non-trivial logic that would otherwise be re-derived each call. A "Workspace Recap" surfaces the agent's self-authored modules in its system message so it remembers what it's built.

**Skills.** Where registration tells the agent *what* it can use, skills tell it *how* to use it effectively. `agent.skill(...)` mounts markdown documentation that the agent reads on-demand — useful for libraries with non-obvious APIs.

**Time-travel debugging.** Every action commits a checkpoint to a kvgit-backed state store. You can pull up the agent's workspace at any past commit:

```python
from agex import events, ActionEvent

action = next(e for e in events(state) if isinstance(e, ActionEvent))
historical = state.checkout(action.commit_hash)
```

## How agex relates to other agent frameworks

There are a lot of agent frameworks. A few rough comparisons in case it helps situate the project:

**JSON-tool frameworks** (LangChain, CrewAI, Pydantic AI): the agent picks from a JSON-typed tool list; arguments serialize across the boundary on each call. agex doesn't have that boundary — your registered modules are the API the agent uses, and rich types pass through without wrapping.

**Shell-based code agents** (Claude Code, Codex CLI, Aider): same general harness shape (stateless code execution + filesystem-as-state), different contract. They're conversational tools; agex's surface is a typed function you call from your application.

**[smolagents](https://github.com/huggingface/smolagents)**: the closest cousin. Same core thesis — agents that think in code instead of choosing tools. smolagents is shaped as a standalone agent product; agex is shaped as an embeddable library. smolagents has more momentum and contributors; if you want a working code-thinking agent today, smolagents is the safer pick. agex explores the embedded-library shape, with stricter typed contracts and a sandbox that runs in the browser.

## The result

agex's surface is a small set of ideas with predictable corollaries:

- The contract is a typed Python function.
- The action space is sandboxed Python over your registered modules.
- The state model is the filesystem, a session cache, and an event log, all versioned.
- The runtime is wherever Python runs, including a browser tab.

Multi-agent workflows become Python control flow. Data handoffs become object passing. Capabilities become library registrations. There's no extra layer — agex reuses the parts of Python that already work.
