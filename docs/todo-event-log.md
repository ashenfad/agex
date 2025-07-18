# The Unified Event Log

## The Goal

To move towards the vision in `todo-streaming.md`, we must first refactor the agent's internal logging mechanism. We will replace the disparate `__stdout__` and `__msg_log__` state variables with a single, unified `__event_log__`.

This log will be a chronological, high-fidelity, pre-render record of everything that happens during a task's execution. It will serve as the single source of truth for two critical functions:
1.  **Agent Context**: Generating the `user` and `assistant` messages for the LLM.
2.  **Developer Streaming**: Providing the event stream for live inspection and post-hoc replay.

## Core Architecture

### 1. The `__event_log__`
The agent's state will now contain a single list named `__event_log__`. This list will hold instances of the Pydantic `*Event` models defined in `agex/agent/events.py` (`TaskStartEvent`, `ActionEvent`, `OutputEvent`, etc.).

### 2. The "Store-Aware" Builtins
A critical design decision is that event-producing builtins like `print` and `view_image` must behave differently depending on the agent's state backing store. This is to ensure the `__event_log__` remains a functionally immutable record of what was observed at a specific point in time.

#### Behavior in `Ephemeral` Mode
-   **Problem**: The `Ephemeral` store holds live, mutable Python objects by reference. If an object is logged and then mutated in the same turn, the log would reflect the final mutated state, which is incorrect.
-   **Solution**: When backed by `Ephemeral` state, the `print` builtin (and others like it) will be responsible for "snapshotting" its arguments before creating an `OutputEvent`.
    -   For standard copyable types (`dict`, `list`), it will use `copy.deepcopy()`.
    -   For complex, un-pickleable, or un-copyable types (like a database connection), it will fall back to storing the object's `repr()` string.

#### Behavior in `Versioned` Mode
-   **Problem**: We want to avoid a major refactor of the `Versioned` state system to make it fully content-addressable at this time.
-   **Solution**: When backed by `Versioned` state, the `print` builtin will store a **raw reference** to the live Python object inside the `OutputEvent`. The existing `Versioned.snapshot()` process will then handle serializing this object as part of the event.
    -   **Trade-off**: This leads to a known inefficiency where an object can be stored twice (once for its own state variable and once inside the event log). We accept this as a temporary trade-off to accelerate the implementation of the streaming feature.

### 3. From Log to LLM Context
The process of generating the conversation for the LLM will be changed. Instead of reading a `__msg_log__`, the `conversation_log` function will now read the `__event_log__` and render the events into `Message` objects just-in-time.

To maintain the behavior of only showing the agent the *most recent* output, the renderer will find the last `ActionEvent` in the log and only render the `OutputEvent`s that have occurred since then to create the `user` message for the current turn.

## Path to `todo-streaming.md`
This refactor is the direct prerequisite for implementing the vision in `todo-streaming.md`. Once the `__event_log__` is in place:
-   **`my_task.stream()`**: The live stream will be implemented by simply yielding events from the `__event_log__` as they are appended.
-   **`agex.replay_stream()`**: The replay stream will be implemented by reading the `__event_log__` from a saved `Versioned` state and yielding its events. Because the `Versioned` events contain references to the raw objects, this will allow for the high-fidelity "live object replay" described in the original plan.

By implementing this unified event log first, we create a solid foundation that makes the subsequent streaming implementation dramatically simpler and more robust.

## Future Performance Optimizations

The "render just-in-time" approach is the most flexible, but we anticipate two areas where performance could become a concern in very long-running agent tasks. We have a clear path to address these without changing the core architecture.

-   **Caching Renders**: If rendering the full event log to a list of `Message` objects becomes costly, we can introduce a cache. The rendered `Message` for each event is deterministic, so we can cache the output and only render new events on each turn.

-   **Faded Memory Caching**: This caching strategy is fully compatible with the `faded-memory` concept. When implementing faded memory, we can cache not just the final render, but the token counts for objects at different, pre-defined rendering detail tiers (e.g., `high`, `medium`, `low`). This would allow the faded memory renderer to quickly select the highest-detail representation that fits within its decaying token budget without having to perform expensive tokenization on every object, every turn. 