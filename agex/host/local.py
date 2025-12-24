"""
Local host implementation.

Executes agent tasks in the current process using the agent's task loop.
"""

from typing import TYPE_CHECKING, Any, Callable

from .base import Host

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent


class Local(Host):
    """
    Local execution host.

    Runs agent tasks in the current process. This is the default host
    and provides the same behavior as calling tasks without any host configuration.
    """

    def execute(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        state: Any,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Execute the task locally using the agent's task loop."""
        # Get the registered task
        task_fn = agent._tasks.get(task_name)
        if task_fn is None:
            raise ValueError(f"Task '{task_name}' not found on agent '{agent.name}'")

        # Call the task directly - it handles the loop internally
        call_kwargs = dict(kwargs)
        call_kwargs["state"] = state
        if on_event is not None:
            call_kwargs["on_event"] = on_event
        if on_token is not None:
            call_kwargs["on_token"] = on_token

        return task_fn(*args, **call_kwargs)

    async def aexecute(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        state: Any,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Execute the task locally using the agent's async task loop."""
        # Get the registered task
        task_fn = agent._tasks.get(task_name)
        if task_fn is None:
            raise ValueError(f"Task '{task_name}' not found on agent '{agent.name}'")

        # Call the task directly - it handles the loop internally
        call_kwargs = dict(kwargs)
        call_kwargs["state"] = state
        if on_event is not None:
            call_kwargs["on_event"] = on_event
        if on_token is not None:
            call_kwargs["on_token"] = on_token

        return await task_fn(*args, **call_kwargs)
