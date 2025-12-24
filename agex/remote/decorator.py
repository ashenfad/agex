"""
@remote decorator for remote agent task execution.

This module provides the decorator that wraps agent tasks for remote execution.
The heavy lifting is delegated to RemoteTaskExecutor.
"""

import functools
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.remote.executor import (
    RemoteTaskExecutor,
)


def remote(
    url: str,
    state: str | None = None,
    timeout: float = 300.0,
    retries: int = 0,
) -> Callable:
    """
    Decorator to mark an agent task for remote execution.

    Args:
        url: The remote server URL (e.g., "https://compute.example.com/execute").
        state: Default state URI (e.g., "disk://default"). Can be overridden at call time.
        timeout: Client-side HTTP timeout in seconds.
        retries: Number of connection retries for network failures only.
            Per design: retries apply to connection refused, DNS failure, etc.
            If a request reaches the server, it will NOT be retried to avoid duplicate execution.
    """

    def decorator(task_func: Callable) -> Callable:
        agent = _resolve_agent(task_func)
        executor = RemoteTaskExecutor(
            agent=agent,
            task_name=task_func.__name__,
            url=url,
            default_state=state,
            timeout=timeout,
            retries=retries,
        )

        # Check if original task is async using the marker set by @agent.task
        is_async = getattr(task_func, "__agex_is_async__", False)

        if is_async:
            return _create_async_wrapper(task_func, executor, state)
        else:
            return _create_sync_wrapper(task_func, executor, state)

    return decorator


def _resolve_agent(task_func: Callable) -> BaseAgent:
    """Extract the agent instance from the task wrapper."""
    agent: BaseAgent | None = getattr(task_func, "__agex_agent__", None)

    if agent is None:
        # Fallback: check for task_agent_fingerprint on UserFunction
        fingerprint = getattr(task_func, "task_agent_fingerprint", None)
        if fingerprint:
            from agex.agent.base import resolve_agent as resolve_by_fingerprint

            agent = resolve_by_fingerprint(fingerprint)

    if not isinstance(agent, BaseAgent):
        raise ValueError(
            "@remote must be the outermost decorator, applied to an @agent.task decorated function."
        )

    return agent


def _extract_remote_kwargs(
    kwargs: dict, default_state: str | None
) -> tuple[str | None, Callable | None, Callable | None]:
    """
    Extract remote-specific kwargs (state, on_token, on_event) from the call kwargs.

    Returns (state_uri, on_token, on_event) and modifies kwargs in-place to remove these.
    """
    # Extract state override
    call_state = kwargs.get("state")
    state_uri = None
    if isinstance(call_state, str):
        kwargs.pop("state")
        state_uri = call_state
    elif call_state is None and default_state is not None:
        state_uri = default_state

    # Extract callbacks
    on_token = kwargs.pop("on_token", None)
    on_event = kwargs.pop("on_event", None)

    return state_uri, on_token, on_event


def _create_sync_wrapper(
    task_func: Callable,
    executor: RemoteTaskExecutor,
    default_state: str | None,
) -> Callable:
    """Create synchronous wrapper function."""

    @functools.wraps(task_func)
    def wrapper(*args, **kwargs) -> Any:
        state_uri, on_token, on_event = _extract_remote_kwargs(kwargs, default_state)
        return executor.execute_sync(args, kwargs, state_uri, on_token, on_event)

    return wrapper


def _create_async_wrapper(
    task_func: Callable,
    executor: RemoteTaskExecutor,
    default_state: str | None,
) -> Callable:
    """Create asynchronous wrapper function."""

    @functools.wraps(task_func)
    async def wrapper(*args, **kwargs) -> Any:
        state_uri, on_token, on_event = _extract_remote_kwargs(kwargs, default_state)
        return await executor.execute_async(args, kwargs, state_uri, on_token, on_event)

    return wrapper
