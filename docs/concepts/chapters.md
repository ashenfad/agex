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

When context exceeds the `log_high_water_tokens` threshold, the framework triggers a special `__chapter__` task. This is a regular agex task — the agent sees its full event history with `[N]` prefixes and a compact index, then creates `Chapter` instances:

```python
Chapter(start=1, end=4, name="Data exploration", message="Found 3 tables...")
```

The framework converts these to `ChapterEvent` instances, splicing them into the event log. The agent's own chapter task events stay in the log and may themselves be chaptered in future rounds.

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

Enable chaptering by setting water marks on the agent:

```python
agent = Agent(
    llm=connect_llm(provider="anthropic", model="claude-sonnet-4-5"),
    state=connect_state(type="versioned", storage="memory"),
    log_high_water_tokens=100_000,   # Trigger chaptering above this
    log_low_water_tokens=50_000,     # Stop chaptering below this
)
```

| Parameter | Description |
|---|---|
| `log_high_water_tokens` | Chaptering triggers when the most recent LLM call's `input_tokens` exceeds this value. |
| `log_low_water_tokens` | Chaptering stops when `input_tokens` drops below this. Defaults to 50% of the high water mark. |

When these are set, the framework automatically:

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

Nothing is deleted. `ChapterEvent.events` holds the originals, and the VFS makes them browsable. An agent that needs a specific detail from hours ago can find it.

### Incremental Rounds

Chaptering runs up to 3 rounds per trigger. Each round:

1. Checks if `input_tokens` still exceeds the high water mark
2. Builds a numbered event index for the agent
3. Calls the `__chapter__` task
4. Applies the returned chapters to the event log
5. Checks if `input_tokens` has dropped below the low water mark

This handles the case where one round of chaptering isn't enough to get below the threshold.

## Event Numbering

All events in the agent's context carry `[N]` prefixes:

```
[1] Task: "Analyze the dataset"
[2] Action: Load and inspect (8 lines)
[3] Output: DataFrame with 12,450 rows...
[4] Chapter: "Data exploration" — Found 3 tables, 3% nulls
[5] Action: Build pipeline (15 lines)
```

These numbers are always visible (not just during chaptering) and correspond to the indices in the chapter task's event index. This gives agents a consistent mental model of their event history.
