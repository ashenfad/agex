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
from agex.llm.core import Message
from agex.render.definitions import render_definitions
from agex.render.value import ValueRenderer

from ..eval.core import evaluate_program
from ..render.context import ContextRenderer
from ..state import Namespaced, Versioned
from ..state.kv import Memory


class TaskLoopMixin(BaseAgent):
    def _run_task_loop(
        self,
        task_name: str,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
        state: Versioned | Namespaced | None,
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
        # Determine state and versioning responsibility
        versioned_state: Versioned | None = None
        if isinstance(state, Namespaced):
            # Namespaced = someone else owns versioning, we just work within namespace
            exec_state = state
            versioned_state = None
        elif isinstance(state, Versioned):
            # Versioned = we're responsible for versioning this state
            versioned_state = state
            exec_state = Namespaced(versioned_state, namespace=self.name)
        else:
            # None = we create and own new versioned state
            versioned_state = Versioned(Memory())
            exec_state = Namespaced(versioned_state, namespace=self.name)

        # Add inputs and expected return type to state for agent access
        if inputs_instance is not None:
            exec_state.set("inputs", inputs_instance)
        exec_state.set("__expected_return_type__", return_type)

        # Initialize conversation log
        initialize_conversation_log(exec_state)

        # Use the agent's configured model for context rendering
        context_renderer = ContextRenderer(self.llm_config["model"])

        # Build system message (always static, never stored in state)
        system_message = self._build_system_message()

        # Add task message to conversation log for this invocation
        initial_task_message = self._build_task_message(
            docstring, inputs_dataclass, inputs_instance, return_type
        )
        add_message(exec_state, Message(role="user", content=initial_task_message))

        # Main task loop
        for iteration in range(self.max_iterations):
            # Clear ALL stdout at the beginning of each iteration so only recent output is shown
            exec_state.set("__stdout__", [])

            # Reconstruct conversation from state
            messages = conversation_log(exec_state, system_message)

            print("ADAM ------- tail of log....")
            print(messages[-1])

            # Get LLM response and determine what code to evaluate

            # Try to get structured response first
            llm_response = self._get_llm_response(messages)
            code_to_evaluate = llm_response.code

            print("ADAM ----------------- think:")
            print(llm_response.thinking)
            print("ADAM ----------------- code:")
            print(llm_response.code)
            print("----------------------------")

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
                # Catch evaluation errors and put them on stdout for agent feedback FIRST
                current_stdout = exec_state.get("__stdout__", [])
                current_stdout.append(f"💥 Evaluation error: {e}")
                exec_state.set("__stdout__", current_stdout)

                # THEN render context (which will now include the error)
                current_context = context_renderer.render(exec_state, self.max_tokens)
                markdown_context = format_context_as_markdown(current_context)
                add_message(
                    exec_state, Message(role="system", content=markdown_context)
                )
            finally:
                # Always snapshot after each evaluation iteration (if we own the state)
                if versioned_state is not None:
                    result = versioned_state.snapshot()
                    if result.unsaved_keys:
                        # Add a message to stdout about the unsaved keys
                        current_stdout = exec_state.get("__stdout__", [])
                        current_stdout.append(
                            f"⚠️ Could not save the following variables because they "
                            f"are not serializable: {', '.join(result.unsaved_keys)}"
                        )
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
            # Use a ValueRenderer with a much longer limit for the initial display.
            # This renderer is only used for this initial task message.
            task_input_renderer = ValueRenderer(max_len=2048, max_depth=4)
            rendered_inputs_value = task_input_renderer.render(inputs_instance)
            rendered_inputs = f"inputs = {rendered_inputs_value}"

            parts.append(
                "Details for your task are available in the `inputs` variable. "
                "Here is its structure and content:"
            )
            parts.append(f"```\n{rendered_inputs}\n```")

        # Add expected output format with clarification for function types
        return_type_name = getattr(return_type, "__name__", str(return_type))

        if return_type_name == "Callable" or "Callable" in str(return_type):
            parts.append(
                f"When complete, call `exit_success(your_function)` where your_function is the {return_type_name} you created. "
                "Pass the function object itself, not the result of calling the function."
            )
        else:
            parts.append(
                f"When complete, call `exit_success(result: {return_type_name})` with your result."
            )

        return "\n\n".join(parts)

    def _get_llm_response(self, messages):
        """Get structured response from the agent's configured LLM client."""
        return self.llm_client.complete(messages)
