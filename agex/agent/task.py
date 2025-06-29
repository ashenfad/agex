"""
Task decorator mixin for Agent class.

This module provides the TaskMixin that handles the @agent.task decorator
which wraps functions to become agent tasks.
"""

import inspect
from dataclasses import make_dataclass
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.agent.loop import TaskLoopMixin
from agex.agent.utils import is_function_body_empty
from agex.eval.validation import validate_with_sampling

# Global registry for dynamically created input dataclasses
# This allows pickle to find them by module.classname lookup
_DYNAMIC_DATACLASS_REGISTRY: dict[str, type] = {}


def clear_dynamic_dataclass_registry() -> None:
    """Clear the dynamic dataclass registry. Useful for testing or memory management."""
    global _DYNAMIC_DATACLASS_REGISTRY
    # Remove from module globals
    for class_name in list(_DYNAMIC_DATACLASS_REGISTRY.keys()):
        globals().pop(class_name, None)
    # Clear the registry
    _DYNAMIC_DATACLASS_REGISTRY.clear()


class TaskMixin(TaskLoopMixin, BaseAgent):
    def task(self, primer_or_func=None, /, *, primer: str | None = None) -> Callable:
        """
        Decorator to mark a function as an agent task.

        The decorated function must have an empty body (only pass, docstrings, comments).
        The decorator replaces the function with one that triggers the agent's task loop.

        Usage:
            # Naked decorator - uses docstring for agent instructions
            @agent.task
            def my_function():
                '''Clear instructions for both agent and caller.'''
                pass

            # Parameterized with no args - same as naked
            @agent.task()
            def my_function():
                '''Clear instructions for both agent and caller.'''
                pass

            # Parameterized with primer - primer for agent, docstring for caller
            @agent.task("Detailed agent implementation instructions")
            def my_function():
                '''Public API documentation for callers.'''
                pass

        Args:
            primer_or_func: Either the primer string or the function being decorated
            primer: Keyword-only primer argument (alternative to positional)

        Returns:
            Either the decorated function (naked) or a decorator function (parameterized)
        """
        # Naked decorator: @agent.task
        if callable(primer_or_func):
            func = primer_or_func
            self._validate_task_decorator(func)
            return self._create_task_wrapper(func, primer=None)

        # Parameterized decorator: @agent.task() or @agent.task("primer")
        def decorator(func: Callable) -> Callable:
            self._validate_task_decorator(func)
            effective_primer = primer_or_func or primer
            return self._create_task_wrapper(func, primer=effective_primer)

        return decorator

    def _validate_task_decorator(self, func: Callable) -> None:
        """Validate that task decorator is being used correctly."""
        # 1. Prevent multiple task decorators (no multi-agent tasks)
        if hasattr(func, "__agex_task_namespace__"):
            existing_namespace = func.__agex_task_namespace__
            raise ValueError(
                f"Function '{func.__name__}' already has a task decorator (namespace: '{existing_namespace}'). "
                f"Multi-agent tasks are not supported."
            )

        # 2. Prevent wrong decorator order (fn must be outer)
        if hasattr(func, "__is_agent_fn__"):
            raise ValueError(
                f"Invalid decorator order on '{func.__name__}'. "
                f"@agent.fn() must be applied AFTER @agent.task(), not before.\n"
                f"Correct order:\n"
                f"@agent.fn()\n"
                f"@agent.task('...')\n"
                f"def {func.__name__}(): ..."
            )

    def _create_task_wrapper(self, func: Callable, primer: str | None) -> Callable:
        """
        Creates the actual task wrapper function.

        Args:
            func: The original function to wrap
            primer: Agent instructions for implementing the task (None to use docstring)

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

        # Determine effective agent instructions
        if primer is not None:
            # Use provided primer for agent instructions
            effective_docstring = primer
        else:
            # Fall back to function docstring
            if func.__doc__ is None or func.__doc__.strip() == "":
                raise ValueError(
                    f"Function '{func.__name__}' decorated with @task must have either "
                    "a primer argument or a non-empty docstring to provide agent instructions."
                )
            effective_docstring = func.__doc__.strip()

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
                validated_args = {}
                for name, value in bound_args.arguments.items():
                    annotation = original_sig.parameters[name].annotation
                    if annotation == inspect.Parameter.empty:
                        annotation = Any  # Default to Any if no type hint
                    try:
                        validated_value = validate_with_sampling(value, annotation)
                        validated_args[name] = validated_value
                    except Exception as e:
                        raise ValueError(
                            f"Validation failed for argument '{name}':\n{e}"
                        ) from e
                inputs_instance = inputs_dataclass(**validated_args)

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

        # Set namespace for dual-decorator pattern (also serves as task-decorated marker)
        namespace = self.name if self.name is not None else self.__class__.__name__
        task_wrapper.__agex_task_namespace__ = namespace

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

        # Make the dataclass pickleable by registering it in module globals
        # This allows pickle to find it via module.classname lookup
        inputs_dataclass.__module__ = __name__  # Set to this module
        _DYNAMIC_DATACLASS_REGISTRY[dataclass_name] = inputs_dataclass
        globals()[dataclass_name] = inputs_dataclass  # Make it findable by pickle

        # Register the dataclass with the agent for sandbox access
        if hasattr(self, "cls"):
            self.cls(inputs_dataclass, constructable=False)  # type: ignore

        return inputs_dataclass
