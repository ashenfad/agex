# Chapters: Agent-Directed Context Compaction

As agents work through long tasks or multi-task sessions, their context grows. Every action, output, and result accumulates in the event log and eventually presses against the LLM's context window. Chapters solve this by letting **agents decide** what to compact and how to summarize it — keeping full detail where it matters and distilling the rest.

## The Problem with Automated Summarization

A common approach is to summarize old events with an LLM call when context gets large. This works, but has drawbacks:

- **The framework guesses** what's important — it doesn't know what the agent still needs
- **Details are lost** — the summary replaces the originals
- **Timing is rigid** — summarization fires based on token counts, not logical boundaries

Chapters flip the model: the agent is the one who decides what to close out, writes the summary, and keeps active work intact.

## How It Works

### ChapterEvent

A `ChapterEvent` replaces a contiguous range of events in the log with a named summary. The original events are preserved inside the chapter — nothing is lost.

```
📖 Chapter: "Data exploration"

Loaded the CSV (12,450 rows × 8 columns). Found 3% null values
concentrated in the `income` column. Schema: id, name, age, income,
city, state, signup_date, plan_type.

Full details: /chapters/data-exploration/
```

When the agent sees this in context, it gets the summary plus a path to browse the originals if needed.

### The Chapter Task

When context exceeds the `chaptering_trigger` threshold, the framework triggers a special `__chapter__` task. This is a regular agex task — the agent sees its full context with task starts numbered `[N]` and a compact task index, then creates `Chapter` instances that close out entire tasks:

```python
Chapter(start=1, end=3, name="Data exploration", message="Found 3 tables...")
```

Here `start` and `end` are 1-based inclusive task numbers from the task index. The framework converts these to `ChapterEvent` instances, splicing them into the event log. The agent's own chapter task events stay in the log and may themselves be chaptered in future rounds.

### Browsing Chaptered History

Original events are accessible via a read-only VFS overlay at `/chapters`:

```
/chapters/data-exploration/
    summary.md              # Chapter name + message
    events/
        001-taskstart.md    # Original TaskStartEvent
        002-action.md       # Original ActionEvent
        003-output.md       # Original OutputEvent
        004-success.md      # Original SuccessEvent
```

Agents browse these with standard file tools (`ls`, `read`) — no special API needed. Nested chapters recurse naturally into `/chapters/{slug}/chapters/{sub-slug}/`.

## Configuration

Enable chaptering by setting the trigger threshold on the agent:

```python
agent = Agent(
    llm=connect_llm(provider="anthropic", model="claude-sonnet-4-5"),
    state=connect_state(type="versioned", storage="memory"),
    chaptering_trigger=100_000,   # Trigger chaptering above this
)
```

| Parameter | Description |
|---|---|
| `chaptering_trigger` | Chaptering triggers when the most recent LLM call's `input_tokens` exceeds this value. |

When set, the framework automatically:

1. Registers the `Chapter` class so agents can construct instances
2. Registers the `__chapter__` task (async) with a primer explaining the protocol
3. Triggers chaptering between tasks and during long-running tasks (after each `task_continue()`)
4. Mounts the `/chapters` VFS overlay for browsing history

The chapter task is always async, ensuring compatibility with async-only LLM providers (e.g. browser-based environments like Pyodide).

## Design Principles

### Agent Autonomy

The agent chooses *what* to chapter and *how* to summarize. The framework only decides *when* to ask. This means:

- Completed work gets chaptered; active investigations stay in full context
- Summaries capture what the agent considers important, not a generic distillation
- The agent can return an empty list if it decides nothing should be chaptered yet

### Lossless Compaction

Nothing is deleted. `ChapterEvent.event_refs` holds references to the originals in state, and the VFS makes them browsable. An agent that needs a specific detail from hours ago can find it.

### Single Round

When triggered, chaptering runs a single round:

1. Checks if `input_tokens` exceeds the chaptering trigger
2. Builds a numbered task index for the agent
3. Calls the `__chapter__` task
4. Applies the returned chapters to the event log

If context is still above the trigger after one round, chaptering will fire again on the next action.

## Task Numbering

Task starts in the agent's context carry `[N]` prefixes:

```
[1] Analyze the dataset
    <THINKING>...</THINKING>
    <PYTHON>...</PYTHON>
    <OBSERVATION>DataFrame with 12,450 rows...</OBSERVATION>
    <TASK_SUCCESS>Found 3 tables</TASK_SUCCESS>
[2] Build the pipeline
    <THINKING>...</THINKING>
    <PYTHON>...</PYTHON>
```

These numbers are always visible (not just during chaptering) and correspond to the task numbers in the chapter task's index. Only `TaskStartEvent` boundaries are numbered — individual actions, outputs, and results within a task are unnumbered. This gives agents a consistent, task-oriented mental model of their history and ensures chaptering operates at clean task boundaries.
