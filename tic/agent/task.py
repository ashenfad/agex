"""
Task decorator mixin for Agent class.

This module provides the TaskMixin that handles the @agent.task decorator
which wraps functions to become agent tasks.
"""

import inspect
from typing import Callable

from tic.agent.base import BaseAgent
from tic.agent.loop import TaskLoopMixin
from tic.agent.utils import is_function_body_empty


class TaskMixin(TaskLoopMixin, BaseAgent):
    def task(self, func_or_docstring_override=None):
        """
        Decorator to mark a function as an agent task.

        The decorated function must have an empty body (only pass, docstrings, comments).
        The decorator replaces the function with one that triggers the agent's task loop.

        Usage:
            @agent.task
            def my_function():
                pass

            @agent.task("Custom prompt override")
            def my_function():
                pass

        Args:
            func_or_docstring_override: Either a function (when used as @agent.task)
                                       or a string override (when used as @agent.task("..."))

        Returns:
            A decorator function or the decorated function
        """
        # Handle @agent.task (without parentheses)
        if callable(func_or_docstring_override):
            func = func_or_docstring_override
            docstring_override = None
            return self._create_task_wrapper(func, docstring_override)

        # Handle @agent.task("override") (with parentheses)
        else:
            docstring_override = func_or_docstring_override

            def decorator(func: Callable) -> Callable:
                return self._create_task_wrapper(func, docstring_override)

            return decorator

    def _create_task_wrapper(
        self, func: Callable, docstring_override: str | None
    ) -> Callable:
        """
        Creates the actual task wrapper function.

        Args:
            func: The original function to wrap
            docstring_override: Optional docstring override

        Returns:
            The wrapped function
        """
        # Validate that the function body is empty
        if not is_function_body_empty(func):
            raise ValueError(
                f"Function '{func.__name__}' decorated with @task must have an empty body. "
                "The agent will provide the implementation."
            )

        # Capture original function metadata
        original_sig = inspect.signature(func)
        return_type = original_sig.return_annotation
        task_name = func.__name__

        # Use docstring override or original docstring
        effective_docstring = docstring_override or func.__doc__

        # Create new signature with added state parameter
        new_params = list(original_sig.parameters.values())
        state_param = inspect.Parameter(
            "state",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Versioned | None",
        )
        new_params.append(state_param)
        new_sig = original_sig.replace(parameters=new_params)

        # Create the replacement function
        def task_wrapper(*args, **kwargs):
            # Bind the arguments to the original signature (excluding state)
            bound_args = original_sig.bind(
                *args, **{k: v for k, v in kwargs.items() if k != "state"}
            )
            bound_args.apply_defaults()

            # Extract state from kwargs
            state = kwargs.get("state", None)

            # Call the task loop
            return self._run_task_loop(
                task_name=task_name,
                docstring=effective_docstring,
                inputs=dict(bound_args.arguments),
                return_type=return_type,
                state=state,
            )

        # Preserve metadata
        task_wrapper.__name__ = func.__name__
        task_wrapper.__doc__ = func.__doc__
        task_wrapper.__annotations__ = func.__annotations__.copy()
        task_wrapper.__annotations__["state"] = "Versioned | None"
        task_wrapper.__signature__ = new_sig

        return task_wrapper
