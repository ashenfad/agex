"""
Task loop mixin for Agent class.

This module provides the TaskLoopMixin that handles the core think→act loop
for agent tasks, following the same mixin pattern as the evaluator.
"""

import inspect
from typing import Any, Callable, Dict

from tic.agent.base import BaseAgent
from tic.agent.utils import is_function_body_empty
from tic.llm.core import Message

from ..eval.core import evaluate_program
from ..render.context import ContextRenderer
from ..state import Versioned
from ..state.kv import Memory


class TaskLoopMixin(BaseAgent):
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

    def _run_task_loop(
        self,
        task_name: str,
        docstring: str | None,
        inputs: Dict[str, Any],
        return_type: type,
        state: Versioned | None,
    ):
        """
        Execute the agent task loop.

        Args:
            task_name: Name of the task function
            docstring: Task description (prompt for the agent)
            inputs: Bound arguments from the function call
            return_type: Expected return type for validation
            state: Optional persistent state

        Returns:
            The validated result from the agent

        Raises:
            TaskFailedException: If agent calls exit_fail()
            ClarificationRequested: If agent calls exit_clarify()
        """
        # Create state if none provided (memory-backed versioned store)
        exec_state = state if state is not None else Versioned(Memory())

        # TODO: Get model name for ContextRenderer - for now use placeholder
        context_renderer = ContextRenderer("gpt-4")  # TODO: Make configurable

        # Build initial system message
        system_message = self._build_system_message(docstring, inputs)

        # Main task loop
        for iteration in range(self.max_iterations):
            try:
                # Build full prompt using ContextRenderer for current state
                current_context = context_renderer.render(exec_state, self.max_tokens)

                # Build messages for LLM
                messages = [Message(role="system", content=system_message)]
                if current_context.strip():
                    messages.append(Message(role="user", content=current_context))

                # Get LLM response
                agent_response = self._get_llm_response(messages)

                # Evaluate the agent's code
                # Cast self to Agent for the evaluate_program call

                evaluate_program(
                    agent_response,
                    self,  # type: ignore
                    exec_state,
                    self.timeout_seconds,
                )

                # Note: stdout is now updated in exec_state, ContextRenderer will handle it

                # Check for completion (would be handled by exit functions in real implementation)
                # TODO: This is where exit_success/fail/clarify would be caught and handled

            except Exception as e:
                # TODO: Handle different exception types (exit functions, errors, etc.)
                raise NotImplementedError(
                    f"Task loop exception handling not implemented: {e}"
                )

        # If we get here, we hit max iterations
        raise TimeoutError(
            f"Task '{task_name}' exceeded maximum iterations ({self.max_iterations})"
        )

    def _build_system_message(
        self, docstring: str | None, inputs: Dict[str, Any]
    ) -> str:
        """Build the initial system message with primer, task description, and inputs."""
        parts = []

        # Add agent primer if available
        if hasattr(self, "primer") and self.primer:
            parts.append(self.primer)

        # Add builtin primer
        builtin_primer = "You are in a Python REPL environment. Use exit_success(result) when complete."
        parts.append(builtin_primer)

        # Add task description
        if docstring:
            parts.append(f"Task: {docstring}")

        # Add inputs
        parts.append(f"Inputs: {self._format_inputs(inputs)}")

        return "\n\n".join(parts)

    def _get_llm_response(self, messages) -> str:
        """Get response from the agent's configured LLM client."""
        return self.llm_client.complete(messages)

    def _format_inputs(self, inputs: Dict[str, Any]) -> str:
        """Format function inputs for display."""
        if not inputs:
            return "None"

        # TODO: Use render system for smart formatting
        formatted_items = []
        for key, value in inputs.items():
            value_str = repr(value)
            if len(value_str) > 200:
                value_str = value_str[:197] + "..."
            formatted_items.append(f"  {key}: {value_str}")

        return "{\n" + ",\n".join(formatted_items) + "\n}"
