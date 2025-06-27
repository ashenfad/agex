"""
Task loop execution mixin for Agent class.

This module provides the TaskLoopMixin that handles the core think→act loop
for agent tasks, including LLM communication and code evaluation.
"""

from typing import Any

from agex.agent.base import BaseAgent
from agex.agent.conversation import (
    add_message,
    conversation_log,
    initialize_conversation_log,
)
from agex.agent.datatypes import ExitSuccess, _AgentExit
from agex.agent.formatting import format_context_as_markdown
from agex.agent.primer_text import BUILTIN_PRIMER
from agex.llm.core import Message, ResponseParseError
from agex.render.definitions import render_definitions
from agex.render.stream import StreamRenderer

from ..eval.core import evaluate_program
from ..render.context import ContextRenderer
from ..state import Versioned
from ..state.kv import Memory

# Format guidance message for when agents produce malformed responses
FORMAT_GUIDANCE_TEMPLATE = """⚠️  Response format error: {error}

Please structure your response as:

# Thinking
[your reasoning here]

```python
[your code here]
```"""


class TaskLoopMixin(BaseAgent):
    def _run_task_loop(
        self,
        task_name: str,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
        state: Versioned | None,
    ):
        """
        Execute the agent task loop.

        Args:
            task_name: Name of the task function
            docstring: Task description (prompt for the agent)
            inputs_dataclass: Dynamically created dataclass type for inputs
            inputs_instance: Instance of the inputs dataclass with actual values
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

        # Add inputs to state for agent access
        if inputs_instance is not None:
            exec_state.set("inputs", inputs_instance)

        # Initialize conversation log
        initialize_conversation_log(exec_state)

        # Use the agent's configured model for context rendering
        context_renderer = ContextRenderer(self.llm_config["model"])

        # Build system message (always static, never stored in state)
        system_message = self._build_system_message()

        # Add initial task message to conversation log (first iteration only)
        if not exec_state.get("__msg_log__"):
            initial_task_message = self._build_task_message(
                docstring, inputs_dataclass, inputs_instance, return_type
            )
            add_message(exec_state, Message(role="user", content=initial_task_message))

        # Main task loop
        for iteration in range(self.max_iterations):
            # Reconstruct conversation from state
            messages = conversation_log(exec_state, system_message)

            # Get LLM response and determine what code to evaluate
            code_to_evaluate = None
            llm_response = None

            try:
                # Try to get structured response first
                llm_response = self._get_llm_response(messages)
                code_to_evaluate = llm_response.code

            except ResponseParseError as e:
                # This can happen if the LLM fails to produce a valid structured response.
                # In this case, we don't have a structured object, so we'll log the error
                # for the agent to see and proceed. The code to evaluate will be empty.
                code_to_evaluate = ""
                # Add gentle scolding message to stdout for agent to see
                current_stdout = exec_state.get("__stdout__", [])
                current_stdout.append(FORMAT_GUIDANCE_TEMPLATE.format(error=e))
                exec_state.set("__stdout__", current_stdout)

            # Store assistant response in conversation log
            if llm_response:
                # Serialize the structured response for the log
                assistant_content = (
                    f"# Thinking\n{llm_response.thinking}\n\n"
                    f"```python\n{llm_response.code}\n```"
                )
                add_message(
                    exec_state, Message(role="assistant", content=assistant_content)
                )

            # Evaluate the code (either parsed or raw)
            try:
                if code_to_evaluate:
                    evaluate_program(
                        code_to_evaluate,
                        self,  # type: ignore
                        exec_state,
                        self.timeout_seconds,
                    )

                current_context = context_renderer.render(exec_state, self.max_tokens)
                markdown_context = format_context_as_markdown(current_context)
                add_message(
                    exec_state, Message(role="system", content=markdown_context)
                )

            except ExitSuccess as exit_signal:
                # Task completed successfully - return the result
                return exit_signal.result
            except _AgentExit:
                # Let other agent exit signals pass through (ExitFail, ExitClarify)
                raise
            except Exception as e:
                current_context = context_renderer.render(exec_state, self.max_tokens)
                markdown_context = format_context_as_markdown(current_context)
                add_message(
                    exec_state, Message(role="system", content=markdown_context)
                )

                # Catch evaluation errors and put them on stdout for agent feedback
                current_stdout = exec_state.get("__stdout__", [])
                current_stdout.append(f"💥 Evaluation error: {e}")
                exec_state.set("__stdout__", current_stdout)

        # If we get here, we hit max iterations
        raise TimeoutError(
            f"Task '{task_name}' exceeded maximum iterations ({self.max_iterations})"
        )

    def _build_system_message(self) -> str:
        """Build the system message with builtin primer, registered resources, and agent primer."""
        parts = []

        # Add builtin primer first (foundation)
        parts.append(BUILTIN_PRIMER)

        # Add registered resources (available tools)

        registered_definitions = render_definitions(self)  # type: ignore
        if registered_definitions.strip():
            parts.append("# Registered Resources\n\n" + registered_definitions)

        # Add agent primer if available (specialization)
        if self.primer:
            parts.append(self.primer)

        return "\n\n".join(parts)

    def _build_task_message(
        self,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
    ) -> str:
        """Build the initial user message with task description."""
        parts = []

        # Add task description
        if docstring:
            parts.append(f"Task: {docstring}")
        else:
            parts.append("Please complete the assigned task.")

        # Add note about inputs if they exist
        if inputs_instance is not None:
            renderer = StreamRenderer(model_name=self.llm_config["model"])
            rendered_inputs = renderer.render_state_stream(
                items={"inputs": inputs_instance}, budget=4000
            )
            parts.append(
                "Details for your task are available in the `inputs` variable. "
                "Here is its structure and content:"
            )
            parts.append(f"```\n{rendered_inputs}\n```")

        # Add expected output format
        return_type_name = getattr(return_type, "__name__", str(return_type))
        parts.append(
            f"When complete, call `exit_success(result: {return_type_name})` with your result."
        )

        return "\n\n".join(parts)

    def _get_llm_response(self, messages):
        """Get structured response from the agent's configured LLM client."""
        return self.llm_client.complete(messages)
