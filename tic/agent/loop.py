"""
Task loop execution mixin for Agent class.

This module provides the TaskLoopMixin that handles the core think→act loop
for agent tasks, including LLM communication and code evaluation.
"""

from typing import Any, Dict

from tic.agent.base import BaseAgent
from tic.agent.primer_text import BUILTIN_PRIMER
from tic.llm.core import Message

from ..eval.core import evaluate_program
from ..render.context import ContextRenderer
from ..state import Versioned
from ..state.kv import Memory


class TaskLoopMixin(BaseAgent):
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

        # Use the agent's configured model for context rendering
        context_renderer = ContextRenderer(self.llm_config["model"])

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

        # Add builtin primer from string constant
        parts.append(BUILTIN_PRIMER)

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
