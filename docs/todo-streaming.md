# Agent Introspection: Streaming and Replay

## The Goal

The goal is to provide a comprehensive suite of tools for developers to observe an agent's work, both in real-time and after the fact. This will serve two primary use cases:
1.  **Interactive Inspection**: Allow developers in notebook environments (like Jupyter) to get a rich, real-time transcript of an agent's execution.
2.  **Application Integration**: Provide a simple way to power web UIs or other applications by streaming agent events over a network.

## The Architecture: "Thin Stream" Core + Rich Display & Serialization Layers

We will implement a flexible, layered architecture that provides the right tool for each use case without adding unnecessary complexity to the framework's core.

### 1. The Core Agent REPL: A Smarter `repr`

At the most fundamental level, we will improve how the agent "sees" the world. When an object is produced in the agent's sandboxed REPL, we will generate the most informative string representation possible for the agent's context. This is achieved via a "Representation Cascade":

1.  **If `_repr_markdown_()` exists**: Use it. This is the ideal format for an LLM.
2.  **Else if `_repr_html_()` exists**: This is a signal of a high-quality object. Use the standard `repr()`.
3.  **Else**: Fall back to our custom "shape-summary" renderer to handle large, generic containers (`dict`, `list`) gracefully without data-dumping.

This ensures the agent gets the best possible view of any object, leveraging the rich ecosystem of Python libraries while providing a safety net for built-in types.

### 2. Live Inspection: The "Thin" Raw Event Stream

The base `.stream()` method on an `@agent.task` will provide a high-fidelity stream for in-process inspection.
-   **`my_task.stream(*args, **kwargs)`**: Returns a generator that yields a **stream of mixed types**.
    - For framework events (like an agent's thoughts), it will yield structured Pydantic event models (e.g., `ActionEvent`).
    - For agent-produced outputs (from `print()` or `view_image()`), it will yield the **raw, live Python objects** (e.g., a `DataFrame`, a `matplotlib.Figure`).

This "thin stream" is the source of truth, providing maximum fidelity for in-process consumers.

#### Raw Stream Event Models

The raw stream will yield instances of the following Pydantic models, in addition to raw Python objects for agent outputs.

-   **`TaskStartEvent(BaseModel)`**: Fired once at the beginning of a task.
    -   `agent_name: str`
    -   `task_name: str`
    -   `inputs: dict[str, Any]`

-   **`ActionEvent(BaseModel)`**: Fired when the agent decides on its next thought and code.
    -   `agent_name: str`
    -   `thinking: str`
    -   `code: str`

-   **`OutputEvent(BaseModel)`**: A container for objects produced by the agent's code.
    -   `agent_name: str`
    -   `parts: list[Any]` (A list of the raw Python objects).

-   **`ErrorEvent(BaseModel)`**: Fired for fatal, unrecoverable errors.
    -   `agent_name: str`
    -   `error_message: str`

-   **`SuccessEvent(BaseModel)`**: Fired when the task completes successfully.
    -   `agent_name: str`
    -   `result: Any` (The final return value).

-   **`FailEvent(BaseModel)`**: Fired when the task is explicitly failed.
    -   `agent_name: str`
    -   `message: str`

### 3. The Notebook Experience: Self-Rendering Events

To make the raw stream immediately useful in a notebook, we will make it "self-rendering" by leveraging the **IPython Display Protocol**.

-   **`_repr_markdown_` for Framework Events**: Our event models (`TaskStartEvent`, `ActionEvent`, `SuccessEvent`, etc.) will implement a `_repr_markdown_()` method. This will generate a clean, readable Markdown representation of the event.
-   **Custom Formatter for `OutputEvent`**: For our special `OutputEvent` container, which holds the agent's output, we will register a custom IPython formatter. This formatter will print a clear header with the agent's name and then ask IPython to `display()` each raw object from the event's `parts` list. This ensures every object is rendered using its own richest possible representation.

The result is a seamless and delightful notebook experience. A developer can simply run `for event in my_task.stream(...): display(event)` to get a beautiful, real-time, mixed-media transcript of the agent's work.

### 4. Application Integration: The `mime_stream` Helper

To serve remote clients (like a web UI) that cannot handle live Python objects, we will provide an optional helper function.

-   **`agex.streams.mime_stream(raw_stream)`**: This function consumes a raw stream and yields a new stream that is fully **JSON-serializable and portable**.
    - It converts all framework events into simple dictionaries.
    - For each raw Python object, it uses the IPython Display Protocol (`_repr_html_`, `_repr_png_`, etc.) to create a `DisplayDataPart` dictionary containing a set of standard MIME type representations.

This allows a developer to power a web UI with a single line of code: `portable_stream = agex.streams.mime_stream(my_task.stream(...))`.

### 5. Post-Hoc Analysis: Replaying History from `Versioned` State

To complete the introspection story, we will allow developers to replay an agent's run from its `Versioned` state.

-   **`agex.replay_stream(state, ...)`**: This helper function takes a `Versioned` state object and reconstructs the "thin" raw event stream by analyzing the state's commit history.

This creates a powerful symmetry. The exact same display tools can be used on both a live stream and a replayed stream, decoupling inspection from execution and making the `Versioned` object a complete, replayable audit trail. 