# The Big Picture

Most agent frameworks ask you to define tools. This means JSON schemas wrapping your code and the agent picking from a list and arguments serializing back and forth across the boundary on each call. `agex` doesn't have that boundary. You define a typed Python function with `@task`, and the agent fills it in by writing sandboxed Python that calls into the modules you've registered. Real Python objects (DataFrames, Plotly figures, your Pydantic models) flow back into your code unchanged. Your existing codebase *is* the toolset.

Three structural choices follow from that - a typed-function contract, a library shape, and a pure-Python sandbox that runs anywhere Python runs.

## Code as the medium

Agents don't choose between "using tools" and "writing code." In agex, code is always central:

- Returning a result: `task_success(...)`
- Calling a function: just call it
- Building data structures: native Python syntax
- Debugging: `print()` and read the output next turn
- Creating reusable logic: define helpers and drop them in `helpers/`

Agents operate in a **generate → execute → observe** loop:

1. **Generate** - the LLM writes a block of Python based on the task and registered capabilities.
2. **Execute** - the framework runs the block in a sandbox.
3. **Observe** - output (prints, errors, return values) flows back as the next turn's context.

Errors land in stdout the way they would in a normal Python session. The agent sees the traceback, adjusts, tries again. No special "error-handling tool" needed.

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

The line `agent.module(pd)` is registration - the bridge between your codebase and the agent's action space. It does double duty as guidance (you choose which modules and members to expose) and security (the agent can only reach what you registered).

The return type is part of the contract. agex validates the agent's `task_success(...)` value against the annotation; if it doesn't match, the agent sees a type error and tries again. Since task inputs (prompts) can carry rich types and results can be rich types, agents can do their work symbolically with code. They inspect with reprs rather than reading full JSON payloads - an agent can sort, filter, or join a million-row DataFrame without ever loading its contents into the conversation.

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

This is the opposite shape from standalone-agent frameworks (smolagents, Claude Code, etc.), which run as their own processes and communicate via text or files. agex is closer in shape to a typed function library. The agent is something you *call*, not something you converse with.

### 3. Pure-Python sandbox, runs anywhere

The sandbox is implemented as AST rewriting in pure Python ([sandtrap](https://github.com/ashenfad/sandtrap)). The default in-process mode runs entirely inside your Python interpreter. But that AST sandbox can also be nested in a subprocess, in a kernel-isolated worker (seccomp/Landlock/Seatbelt), or (via [Pyodide](https://pyodide.org/)) entirely in a browser tab.

```python
agent = Agent(
    isolation="none",      # in-process (default)
    # isolation="process",   # subprocess
    # isolation="kernel",    # subprocess + seccomp / Landlock / Seatbelt
)
```

[agex-studio](https://agex.studio) is the proof-of-concept: a chat-driven data analysis app with pandas, scikit-learn, plotly, calendar integration, and an interactive app pane. It exists all client-side with no backend. It's the same agex you'd embed in a server-side application; the only difference is the runtime.

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

**Spawning sub-tasks.** The orchestration above is wired by the developer. Agents can *also* initiate it themselves: `spawn` is injected into agent code so the agent can define a typed sub-task and run it on an ephemeral, memoryless clone of itself — useful for generating sample data, drafting an artifact, or researching a point and getting a validated result back.

```python
@spawn.task
def make_svg(prompt: str) -> str:
    """Return SVG markup for a 64x64 tile."""
    pass

svg   = make_svg("a small castle")              # blocks, returns the typed result
tiles = spawn.map(make_svg, ["a", "b", "c"])    # concurrent fan-out
```

The clone inherits the parent's capabilities (and granted [scopes](security.md#capability-scopes-human-in-the-loop)) but none of its memory, cache, or files — data crosses in through the typed argument. The surface mirrors `concurrent.futures` (`submit` / `map` / `.result()`), is always blocking (no `async`/`await` in agent code), and runs on a thread pool bounded by [`Agent(max_spawns=...)`](../api/agent.md). This is the ergonomic successor to the [dogfood pattern](../examples/dogfood.md) for the common ephemeral case.

**Agent-authored libraries.** Agents can write helper modules to the Virtual Filesystem (`helpers/utils.py`) and `import` them in subsequent tasks - useful for non-trivial logic that would otherwise be re-derived each call. A "Workspace Recap" surfaces the agent's self-authored modules in its system message so it remembers what it's built.

**Skills.** Where registration tells the agent *what* it can use, skills tell it *how* to use it effectively. `agent.skill(...)` mounts markdown documentation that the agent reads on-demand - useful for libraries with non-obvious APIs.

**Terminal-shaped tooling.** Most agent capabilities fit the library shape — registered modules and functions the agent calls in Python.  But some don't: compilers, formatters, archive utilities, anything the agent has seen as a CLI invocation in training rather than a Python API.  `agent.terminal(...)` exposes these as commands the agent runs from `terminal_action` blocks, with the same `--help`-and-pipelines idioms agents already know.

```python
@agent.terminal
def esbuild(ctx):
    """Bundle JS source files."""
    ...
```

The library shape stays primary — that's where work finishes, and `task_success` only fires from `python_action`.  The terminal is a secondary surface for tools whose natural interface isn't a Python function: register Python where it's natural, terminal where it's natural, agents reach for whichever the underlying capability is shaped like.

**Time-travel debugging.** Every action commits a checkpoint to a kvgit-backed state store. You can pull up the agent's workspace at any past commit:

```python
from agex import events, ActionEvent

action = next(e for e in events(state) if isinstance(e, ActionEvent))
historical = state.checkout(action.commit_hash)
```

## How agex relates to other agent frameworks

There are a lot of agent frameworks. A few rough comparisons in case it helps situate the project:

**JSON-tool frameworks** (LangChain, CrewAI, Pydantic AI): the agent picks from a JSON-typed tool list; arguments serialize across the boundary on each call. agex doesn't have that boundary - your registered modules are the API the agent uses, and rich types pass through without wrapping.

**Shell-based code agents** (Claude Code, Codex CLI, Aider): same general harness shape (stateless code execution + filesystem-as-state), different contract. They're conversational tools; agex's surface is a typed function you call from your application.

**[smolagents](https://github.com/huggingface/smolagents)**: the closest cousin. Same core thesis - agents that think in code instead of choosing tools. smolagents is shaped as a standalone agent product; agex is shaped as an embeddable library. smolagents has more momentum and contributors; if you want a working code-thinking agent today, smolagents is the safer pick. agex explores the embedded-library shape, with stricter typed contracts and a sandbox that runs in the browser.

## The result

agex's surface is a small set of ideas with predictable corollaries:

- The contract is a typed Python function.
- The action space is sandboxed Python over your registered modules.
- The state model is the filesystem, a session cache, and an event log, all versioned.
- The runtime is wherever Python runs, including a browser tab.

Multi-agent workflows become Python control flow. Data handoffs become object passing. Capabilities become library registrations. There's no extra layer - agex reuses the parts of Python that already work.
