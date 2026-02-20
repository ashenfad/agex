"""
Helper functions for remote execution server.

This module provides utilities for:
- Bridging sync task execution to async SSE streams
- Formatting events for Server-Sent Events protocol
"""

import base64
import json
from typing import Any

import cloudpickle
from kvit import Store


def execute_worker(
    task_func: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    state: Store | None,
    queue: Any,
    loop: Any,
    finished_sentinel: Any,
) -> None:
    """
    Execute the task in a separate thread and bridge events/results to the async queue.

    This function runs in a thread pool and uses loop.call_soon_threadsafe to
    safely send events back to the async queue.
    """
    try:
        # Prepare kwargs
        exec_kwargs = dict(kwargs)

        # Wire up event handlers
        def on_token(token_chunk):
            loop.call_soon_threadsafe(queue.put_nowait, ("token", token_chunk))

        def on_event(event):
            loop.call_soon_threadsafe(queue.put_nowait, event)

        exec_kwargs["on_event"] = on_event
        exec_kwargs["on_token"] = on_token

        # Execute task
        result = task_func(*args, **exec_kwargs)

        # Send result
        loop.call_soon_threadsafe(queue.put_nowait, (finished_sentinel, result))

    except Exception as e:
        # Capture exception and send as result
        loop.call_soon_threadsafe(queue.put_nowait, (finished_sentinel, e))


async def stream_execution_results(
    queue: Any,
    finished_sentinel: Any,
) -> Any:
    """
    Stream events from the queue until the finished sentinel is received.

    Yields raw formatted SSE strings for StreamingResponse.
    """
    import inspect

    while True:
        item = await queue.get()

        # Check for completion
        if isinstance(item, tuple) and item[0] is finished_sentinel:
            result = item[1]
            if isinstance(result, Exception):
                # Re-raise to be handled by outer try/except (which formats error event)
                raise result

            # Handle async results (if task_func returned coroutine)
            if inspect.isawaitable(result):
                result = await result

            yield format_result_sse(result)
            break

        # Handle tokens
        elif isinstance(item, tuple) and item[0] == "token":
            yield format_token_sse(item[1])

        # Handle normal events
        else:
            yield format_event_sse(item)


# =============================================================================
# Raw Data Formatters (for EventSourceResponse)
# =============================================================================


def format_result_data(result: Any) -> str:
    """Return JSON string for a result event (no SSE framing)."""
    payload = base64.b64encode(cloudpickle.dumps(result)).decode("utf-8")
    return json.dumps({"type": "result", "payload": payload})


def format_error_data(message: str, tb: str | None = None) -> str:
    """Return JSON string for an error event (no SSE framing)."""
    return json.dumps({"type": "error", "message": message, "traceback": tb})


def format_token_data(token_chunk: Any) -> str:
    """Return JSON string for a token event (no SSE framing)."""
    payload = base64.b64encode(cloudpickle.dumps(token_chunk)).decode("utf-8")
    return json.dumps({"type": "token", "payload": payload})


def format_event_data(event: Any) -> str:
    """Return JSON string for an agent event (no SSE framing)."""
    payload = base64.b64encode(cloudpickle.dumps(event)).decode("utf-8")
    return json.dumps({"type": "event", "payload": payload})


# =============================================================================
# SSE-Formatted Strings (for StreamingResponse)
# =============================================================================


def format_result_sse(result: Any) -> str:
    """Format a result as a complete SSE event string."""
    return f"data: {format_result_data(result)}\n\n"


def format_error_sse(message: str, tb: str | None = None) -> str:
    """Format an error as a complete SSE event string."""
    return f"data: {format_error_data(message, tb)}\n\n"


def format_token_sse(token_chunk: Any) -> str:
    """Format a token chunk as a complete SSE event string."""
    return f"data: {format_token_data(token_chunk)}\n\n"


def format_event_sse(event: Any) -> str:
    """Format an agent event as a complete SSE event string."""
    return f"data: {format_event_data(event)}\n\n"
