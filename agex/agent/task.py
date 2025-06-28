"""
Task decorator mixin for Agent class.

This module provides the TaskMixin that handles the @agent.task decorator
which wraps functions to become agent tasks.
"""

import inspect
import pickle
from dataclasses import make_dataclass
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.agent.loop import TaskLoopMixin
from agex.agent.utils import is_function_body_empty


class TaskMixin(TaskLoopMixin, BaseAgent):
    def task(self, primer: str, /) -> Callable:
        """
        Decorator to mark a function as an agent task with required primer.

        The decorated function must have an empty body (only pass, docstrings, comments).
        The decorator replaces the function with one that triggers the agent's task loop.

        Usage:
            @agent.task("Build a function from the given prompt")
            def my_function():
                pass

        Args:
            primer: Instructions for the agent on how to implement this task

        Returns:
            A decorator function
        """

        def decorator(func: Callable) -> Callable:
            return self._create_task_wrapper(func, primer)

        return decorator

    def _create_task_wrapper(self, func: Callable, primer: str) -> Callable:
        """
        Creates the actual task wrapper function.

        Args:
            func: The original function to wrap
            primer: Agent instructions for implementing the task

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

        # Use primer for agent instructions
        effective_docstring = primer

        # Create dynamic dataclass for inputs
        inputs_dataclass = self._create_inputs_dataclass(task_name, original_sig)

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

            # Create inputs dataclass instance with pass-by-value semantics
            inputs_instance = None
            if bound_args.arguments:
                # Force pass-by-value using pickle/unpickle to ensure complete isolation
                try:
                    isolated_args = pickle.loads(pickle.dumps(bound_args.arguments))
                    inputs_instance = inputs_dataclass(**isolated_args)
                except (pickle.PicklingError, TypeError) as e:
                    raise ValueError(
                        f"Task arguments must be serializable for security isolation. "
                        f"Failed to serialize argument: {e}"
                    )

            # Call the task loop
            return self._run_task_loop(
                task_name=task_name,
                docstring=effective_docstring,
                inputs_dataclass=inputs_dataclass,
                inputs_instance=inputs_instance,
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

    def _create_inputs_dataclass(self, task_name: str, signature: inspect.Signature):
        """
        Create a dynamic dataclass for the task inputs.

        Args:
            task_name: Name of the task function
            signature: Function signature to extract parameters from

        Returns:
            Dynamically created dataclass type
        """
        if not signature.parameters:
            # No inputs - return a simple empty dataclass
            return make_dataclass(f"{task_name.title()}Inputs", [])

        # Build field specifications for make_dataclass
        fields = []
        for param_name, param in signature.parameters.items():
            # Get type annotation, default to Any if not specified
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )

            # Handle default values
            if param.default != inspect.Parameter.empty:
                # Has default value
                fields.append((param_name, param_type, param.default))
            else:
                # Required parameter
                fields.append((param_name, param_type))

        # Create the dataclass
        dataclass_name = f"{task_name.title()}Inputs"
        inputs_dataclass = make_dataclass(dataclass_name, fields)

        # Register the dataclass with the agent for sandbox access
        if hasattr(self, "cls"):
            self.cls(inputs_dataclass, constructable=False)  # type: ignore

        return inputs_dataclass
