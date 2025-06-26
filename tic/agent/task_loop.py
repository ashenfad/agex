"""
Task loop mixin for Agent class.

This module provides the TaskLoopMixin that handles the core think→act loop
for agent tasks, following the same mixin pattern as the evaluator.
"""

from typing import Any, Dict

from tic.agent.base import BaseAgent

from ..eval.core import evaluate_program
from ..render.context import ContextRenderer
from ..state import Versioned
from ..state.kv import Memory


class TaskLoopMixin(BaseAgent):
    """
    Mixin that provides task loop functionality to Agent class.

    Assumes the class has: primer, timeout_seconds attributes.
    """

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
        max_iterations = 10  # TODO: Make configurable
        for iteration in range(max_iterations):
            try:
                # Build full prompt using ContextRenderer for current state
                budget = 4000  # TODO: Make configurable based on context window
                current_context = context_renderer.render(exec_state, budget)
                full_prompt = f"{system_message}\n\n{current_context}"

                # TODO: Get LLM response - for now use placeholder
                agent_response = self._get_llm_response(full_prompt)

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
            f"Task '{task_name}' exceeded maximum iterations ({max_iterations})"
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

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM - placeholder for now."""
        # TODO: Integrate with actual LLM
        raise NotImplementedError("LLM integration not implemented yet")

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
