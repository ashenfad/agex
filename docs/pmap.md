# `pmap`: Parallel Map for Concurrent Tasks

## Motivation

In many agentic workflows, we need to perform multiple I/O-bound operations concurrently. For example, an agent might need to fetch data from several different APIs, or run multiple sub-agents that perform independent tasks. While our current execution model is single-threaded, we can introduce a `pmap` (parallel map) built-in to support these use cases without the complexity of full `async/await` support.

## Proposed Implementation

We can implement `pmap` as a new built-in function that leverages Python's `concurrent.futures.ThreadPoolExecutor`.

```python
# rough sketch in Python
from concurrent.futures import ThreadPoolExecutor

def pmap(func, iterable):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(func, iterable))
    return results
```

When this is called from within the Tic environment, the `func` would be one of our `UserFunction` objects. Our existing design has some key advantages that make this feasible and safe:

1.  **Isolated State:** Each function call executes in an isolated state scope. When `pmap` calls the `UserFunction` for each item in the iterable, each call will get its own sandboxed state, preventing race conditions or state corruption between concurrent executions.
2.  **Thread-Safety:** The underlying state management (`Versioned`, `Scoped`, `Ephemeral`) is designed to be safe for this kind of concurrency, as each "thread" of execution operates on its own copy-on-write version of the state. The results are then collected back in the main thread.

## Usage Example

An agent could use `pmap` to run a tool on multiple inputs at once:

```python
@agent.task
def research_topics(topics: list[str]) -> list[str]:
    """
    Given a list of topics, research each one and return a summary.

    This uses pmap to run the research in parallel.
    """
    summaries = pmap(self.research_tool, topics)
    return summaries

@agent.tool
def research_tool(topic: str) -> str:
    """A (simulated) tool that does some I/O bound work."""
    # ... implementation of the tool ...
    return f"Summary for {topic}"
```

This provides a powerful concurrency primitive that fits naturally within our existing execution model. 