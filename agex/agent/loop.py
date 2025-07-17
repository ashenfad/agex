"""
Task loop execution mixin for Agent class.

This module provides the TaskLoopMixin that handles the core think→act loop
for agent tasks, including LLM communication and code evaluation.
"""

import inspect
from typing import Any

from agex.agent.base import BaseAgent
from agex.agent.conversation import (
    add_message,
    conversation_log,
    initialize_conversation_log,
)
from agex.agent.datatypes import (
    TaskContinue,
    TaskFail,
    TaskSuccess,
    _AgentExit,
)
from agex.agent.formatting import format_context_as_markdown
from agex.agent.primer_text import BUILTIN_PRIMER
from agex.llm.core import (
    ContentPart,
    ImagePart,
    MultimodalMessage,
    TextMessage,
    TextPart,
)
from agex.render.definitions import render_definitions
from agex.render.value import ValueRenderer

from ..eval.core import evaluate_program
from ..render.context import ContextRenderer
from ..state import Namespaced, Versioned


class TaskLoopMixin(BaseAgent):
    def _render_and_add_context(self, exec_state, context_renderer: ContextRenderer):
        """Helper method to render context and add it as a message to avoid code duplication."""
        # This now returns a list of content parts, which could include images.
        context_parts: list[ContentPart] = context_renderer.render(
            exec_state, self.max_tokens
        )

        if not context_parts:
            return  # Nothing to add

        # Check if the context is purely text or contains images
        has_images = any(isinstance(part, ImagePart) for part in context_parts)

        if has_images:
            # Create a multimodal message if there are images
            message = MultimodalMessage(role="user", content=context_parts)
        else:
            # Otherwise, combine text parts into a single text message for efficiency
            full_text = "\n".join(
                part.text for part in context_parts if isinstance(part, TextPart)
            )
            message = TextMessage(role="user", content=full_text)

        add_message(exec_state, message)

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
            TaskFail: If agent calls task_fail()
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
            # None = we create and own new ephemeral state (no persistence by default)
            from ..state import Ephemeral

            exec_state = Ephemeral()

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
        print("============== INITIAL TASK MESSAGE ===============")
        print(initial_task_message)
        print("===========================================")
        add_message(exec_state, TextMessage(role="user", content=initial_task_message))

        # Main task loop
        for iteration in range(self.max_iterations):
            # Clear ALL stdout at the beginning of each iteration so only recent output is shown
            exec_state.set("__stdout__", [])

            # Reconstruct conversation from state
            messages = conversation_log(exec_state, system_message)

            print("============== LAST MESSAGE ===============")
            print(messages[-1:])
            print("===========================================")

            # Get LLM response and determine what code to evaluate
            # Try to get structured response first
            llm_response = self._get_llm_response(messages)
            code_to_evaluate = llm_response.code

            print("=============== LLM THOUGHT ===============")
            print(llm_response.thinking)
            print("================ LLM CODE =================")
            print(llm_response.code)
            print("===========================================")

            # Store assistant response in conversation log
            if llm_response:
                # Serialize the structured response for the log
                assistant_content = (
                    f"# Thinking\n{llm_response.thinking}\n\n"
                    f"```python\n{llm_response.code}\n```"
                )
                add_message(
                    exec_state, TextMessage(role="assistant", content=assistant_content)
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

                self._render_and_add_context(exec_state, context_renderer)

            except TaskSuccess as task_signal:
                # Task completed successfully - return the result
                return task_signal.result
            except TaskContinue:
                # Agent wants to continue to next iteration - just continue the loop
                self._render_and_add_context(exec_state, context_renderer)
                continue
            except (TaskFail, _AgentExit):
                # Let other agent exit signals pass through (TaskFail)
                raise
            except Exception as e:
                # Catch evaluation errors and put them on stdout for agent feedback FIRST
                current_stdout = exec_state.get("__stdout__", [])
                current_stdout.append(f"💥 Evaluation error: {e}")
                exec_state.set("__stdout__", current_stdout)

                # THEN render context (which will now include the error)
                self._render_and_add_context(exec_state, context_renderer)
            finally:
                # Always snapshot after each evaluation iteration (if we own the state)
                from ..state import is_ephemeral_root

                if versioned_state is not None and not is_ephemeral_root(exec_state):
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
        print("============== REGISTERED DEFINITIONS ===============")
        print(registered_definitions)
        print("===========================================")
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
            parts.append(
                "\nAccess these values with patterns like `inputs.some_attr`\n"
            )

        # Add expected output format with clarification for function types
        if return_type is inspect.Parameter.empty:
            # No return type annotation - just call task_success() with no arguments
            parts.append("When complete, call `task_success()` to indicate completion.")
        elif "Callable" in str(return_type):
            # Function return type - special instructions
            # Clean up the type representation to remove confusing module references
            return_type_str = str(return_type)
            # Remove "typing." prefix but keep the useful type information
            if return_type_str.startswith("typing."):
                return_type_str = return_type_str[7:]  # Remove "typing." prefix

            parts.append(
                f"When complete, call `task_success(your_function)` where your_function is the {return_type_str} you created. "
                "Pass the function object itself, not the result of calling the function.\n"
            )
        else:
            # Regular return type - show the type annotation
            # Use clean type names for all types when possible
            if (
                hasattr(return_type, "__module__")
                and hasattr(return_type, "__name__")
                and not hasattr(return_type, "__origin__")  # Not a generic type
            ):
                # Use the clean class name for simple types (str, int, custom classes)
                return_type_name = return_type.__name__
            else:
                # For generic types (list[int], dict[str, int]) or complex types,
                # use the full string representation to preserve type parameters
                return_type_name = str(return_type)

            parts.append(
                f"When complete, call `task_success(result: {return_type_name})` with your result."
            )

        return "\n\n".join(parts)

    def _get_llm_response(self, messages):
        """Get structured response from the agent's configured LLM client."""
        return self.llm_client.complete(messages)
