"""
Task loop execution mixin for Agent class.

This module provides the TaskLoopMixin that handles the core think→act loop
for agent tasks, including LLM communication and code evaluation.
"""

import re
from copy import deepcopy
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.agent.conversation import (
    conversation_log,
)
from agex.agent.datatypes import (
    LLMFail,
    TaskClarify,
    TaskContinue,
    TaskFail,
    TaskSuccess,
    TaskTimeout,
    _AgentExit,
)
from agex.agent.events import (
    ActionEvent,
    ClarifyEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.agent.primer_text import BUILTIN_PRIMER
from agex.eval.core import evaluate_program
from agex.eval.error import EvalError
from agex.eval.objects import PrintAction
from agex.render.definitions import render_definitions
from agex.state import Live, Namespaced, Versioned, events
from agex.state.log import add_event_to_log


class TaskLoopMixin(BaseAgent):
    @staticmethod
    def _strip_markdown_code_fence(code: str) -> str:
        """
        Remove surrounding ```python ... ``` (or generic ``` ... ```) fences if the entire
        response code is wrapped in a single fenced block.

        This helps when an LLM mistakenly returns markdown-formatted code blocks.
        """
        if not isinstance(code, str):
            return code

        text = code.strip()
        # Quick check for starting and ending fences
        if not text.startswith("```"):
            return code

        # Match an opening fence (optionally with a language tag) and a closing fence at the end
        # Capture the body in between in a non-greedy way
        pattern = r"^```[A-Za-z0-9_+-]*\s*\n([\s\S]*?)\n```\s*$"
        match = re.match(pattern, text)
        if match:
            body = match.group(1)
            return body
        return code

    def _yield_new_events(self, exec_state, events_yielded_count, on_event):
        """
        Helper method to yield new events and return updated count.
        Uses events() to capture hierarchical events including sub-agents.
        """
        from agex.state import events

        all_events = events(exec_state)  # Gets all events including children
        new_events = all_events[events_yielded_count:]
        for event in new_events:
            # The handler has already been called when the event was created.
            # This generator is just for the stream() consumer.
            yield event
        return len(all_events)

    def _task_loop_generator(
        self,
        task_name: str,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
        state: Versioned | Namespaced | None,
        on_event: Callable[[Any], None] | None = None,
        setup: str | None = None,
    ):
        """
        Generator that yields events as they happen during task execution.
        This is the core implementation used by both streaming and regular modes.
        """
        # Determine state and versioning responsibility (same logic as _run_task_loop)
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
            # None = we create and own new live state (no persistence by default)
            exec_state = Namespaced(Live(), namespace=self.name)

        # Add inputs and expected return type to state for agent access
        if inputs_instance is not None:
            exec_state.set("inputs", inputs_instance)
        exec_state.set("__expected_return_type__", return_type)

        # Initialize the event log if it doesn't exist
        if "__event_log__" not in exec_state:
            exec_state.set("__event_log__", [])

        events_yielded = len(events(exec_state))  # type: ignore

        # Build system message (always static, never stored in state)
        system_message = self._build_system_message()

        # Build the initial task message
        initial_task_message = self._build_task_message(
            docstring, inputs_dataclass, inputs_instance, return_type
        )

        # Create comprehensive task start event with message content
        task_start_event = TaskStartEvent(
            agent_name=self.name,
            task_name=task_name,
            inputs={
                f.name: deepcopy(getattr(inputs_instance, f.name))
                for f in inputs_dataclass.__dataclass_fields__.values()
            },
            message=initial_task_message,
        )
        add_event_to_log(exec_state, task_start_event, on_event=on_event)
        yield task_start_event
        events_yielded += 1

        # Execute setup code if provided (doesn't count against iteration limit)
        if setup:
            # Create ActionEvent for setup
            setup_action_event = ActionEvent(
                agent_name=self.name,
                thinking="This code was automatically run to provide context for the task.",
                code=setup,
            )
            add_event_to_log(exec_state, setup_action_event, on_event=on_event)
            yield setup_action_event
            events_yielded += 1

            # Execute the setup code
            try:
                evaluate_program(
                    setup,
                    self,  # type: ignore
                    exec_state,
                    self.timeout_seconds,
                    on_event=on_event,
                )
            except Exception:
                # Setup errors are handled normally - they become ErrorEvents
                # and the agent can see them in their context
                pass

            # Yield any OutputEvents created during setup execution
            events_yielded = yield from self._yield_new_events(
                exec_state, events_yielded, on_event
            )

        # Main task loop
        for iteration in range(self.max_iterations):
            # Reconstruct conversation from state
            messages = conversation_log(exec_state, system_message, self)

            # Get LLM response with built-in retry and event emission
            llm_response = self._get_llm_response(messages, exec_state, on_event)
            # Sanitize common markdown code-fence wrappers if present
            llm_response.code = self._strip_markdown_code_fence(llm_response.code)
            code_to_evaluate = llm_response.code

            # Store assistant response in event log and yield immediately
            if llm_response:
                action_event = ActionEvent(
                    agent_name=self.name,
                    thinking=llm_response.thinking,
                    code=llm_response.code,
                )
                add_event_to_log(exec_state, action_event, on_event=on_event)
                yield action_event
                events_yielded += 1

            # Evaluate the code (either parsed or raw)
            try:
                if code_to_evaluate:
                    evaluate_program(
                        code_to_evaluate,
                        self,  # type: ignore
                        exec_state,
                        self.timeout_seconds,
                        on_event=on_event,
                    )

            except TaskSuccess as task_signal:
                # Before handling completion, yield any evaluation events first
                events_yielded = yield from self._yield_new_events(
                    exec_state, events_yielded, on_event
                )

                # Task completed successfully - log completion event and return the result
                success_event = SuccessEvent(
                    agent_name=self.name,
                    result=task_signal.result,
                )
                add_event_to_log(exec_state, success_event, on_event=on_event)
                yield success_event
                return task_signal.result
            except TaskContinue:
                # Before continuing, yield any evaluation events first
                events_yielded = yield from self._yield_new_events(
                    exec_state, events_yielded, on_event
                )

                # Agent wants to continue to next iteration - just continue the loop
                continue
            except TaskClarify as task_clarify:
                # Before handling clarification, yield any evaluation events first
                events_yielded = yield from self._yield_new_events(
                    exec_state, events_yielded, on_event
                )

                # Log clarification event
                clarify_event = ClarifyEvent(
                    agent_name=self.name,
                    message=task_clarify.message,
                )
                add_event_to_log(exec_state, clarify_event, on_event=on_event)
                yield clarify_event

                # If we're not top-level (have a Namespaced state), convert to EvalError
                if isinstance(state, Namespaced):
                    raise EvalError(
                        f"Sub-agent needs clarification: {task_clarify.message}", None
                    )
                else:
                    # We're top-level, re-raise the TaskClarify
                    raise

            except TaskFail as task_fail:
                # Before handling failure, yield any evaluation events first
                events_yielded = yield from self._yield_new_events(
                    exec_state, events_yielded, on_event
                )

                # Log failure event
                fail_event = FailEvent(
                    agent_name=self.name,
                    message=task_fail.message,
                )
                add_event_to_log(exec_state, fail_event, on_event=on_event)
                yield fail_event

                # If we're not top-level (have a Namespaced state), convert to EvalError
                if isinstance(state, Namespaced):
                    raise EvalError(f"Sub-agent failed: {task_fail.message}", None)
                else:
                    # We're top-level, re-raise the TaskFail
                    raise
            except LLMFail:
                # Emit fatal ErrorEvent already done in _get_llm_response; propagate
                raise
            except _AgentExit:
                # Before handling exit, yield any evaluation events first
                events_yielded = yield from self._yield_new_events(
                    exec_state, events_yielded, on_event
                )

                # Let other agent exit signals pass through (without logging)
                raise
            except Exception as e:
                # Catch evaluation errors and put them in an OutputEvent so the agent can see them
                error_output = OutputEvent(
                    agent_name=self.name,
                    parts=[
                        PrintAction(
                            [
                                f"💥 Evaluation error: {e}\nYou must adjust your code accordingly!"
                            ]
                        )
                    ],
                )
                add_event_to_log(exec_state, error_output, on_event=on_event)
                yield error_output
                events_yielded += 1
            else:
                # Normal completion - yield any evaluation events
                events_yielded = yield from self._yield_new_events(
                    exec_state, events_yielded, on_event
                )

                # Check if the code executed successfully but didn't call any task_* functions
                # This is a common issue with local models that forget to signal completion
                if code_to_evaluate and code_to_evaluate.strip():
                    # Look for task_* function calls in the executed code
                    has_task_call = any(
                        task_func in code_to_evaluate
                        for task_func in [
                            "task_success(",
                            "task_fail(",
                            "task_clarify(",
                            "task_continue(",
                        ]
                    )

                    if not has_task_call:
                        # Provide helpful guidance to the agent
                        guidance_message = (
                            "💡 **Task Control Reminder**: Your code executed successfully, but you need to signal completion.\n\n"
                            "**Next steps:**\n"
                            "• `task_success(result)` - Complete the task with your final answer\n"
                            "• `task_continue(result)` - Observe your work and continue to another REPL iteration\n"
                            "• `task_fail(message)` - If you cannot complete the task\n"
                            "• `task_clarify(message)` - If you need more information\n\n"
                            "Your code ran without errors - now just add the appropriate task control function!"
                        )

                        guidance_output = OutputEvent(
                            agent_name=self.name,
                            parts=[PrintAction([guidance_message])],
                        )
                        add_event_to_log(exec_state, guidance_output, on_event=on_event)
                        yield guidance_output
                        events_yielded += 1
            finally:
                # Always snapshot after each evaluation iteration (if we own the state)
                from ..state import is_live_root

                if versioned_state is not None and not is_live_root(exec_state):
                    result = versioned_state.snapshot()
                    if result.unsaved_keys:
                        # Add a message to stdout about the unsaved keys so the agent can see it
                        # Strip namespace prefix from keys so agent sees clean variable names
                        agent_visible_keys = []
                        namespace_prefix = f"{self.name}/"
                        for key in result.unsaved_keys:
                            if key.startswith(namespace_prefix):
                                agent_visible_keys.append(key[len(namespace_prefix) :])
                            else:
                                agent_visible_keys.append(key)

                        warning_message = (
                            f"⚠️ Could not save the following variables because they "
                            f"are not serializable: {', '.join(agent_visible_keys)}"
                        )
                        warning_output = OutputEvent(
                            agent_name=self.name,
                            parts=[PrintAction([warning_message])],
                        )
                        add_event_to_log(exec_state, warning_output, on_event=on_event)
                        yield warning_output
                        events_yielded += 1

        # Final snapshot before returning, to catch any last-minute mutations
        # or state changes that occurred during the final successful evaluation.
        if versioned_state is not None:
            result = versioned_state.snapshot()
            if result.unsaved_keys:
                agent_visible_keys = []
                namespace_prefix = f"{self.name}/"
                for key in result.unsaved_keys:
                    if key.startswith(namespace_prefix):
                        agent_visible_keys.append(key[len(namespace_prefix) :])
                    else:
                        agent_visible_keys.append(key)

                warning_message = (
                    f"⚠️ Could not save the following variables because they "
                    f"are not serializable: {', '.join(agent_visible_keys)}"
                )
                warning_output = OutputEvent(
                    agent_name=self.name,
                    parts=[PrintAction([warning_message])],
                )
                add_event_to_log(exec_state, warning_output, on_event=on_event)
                yield warning_output
                events_yielded += 1

        # If we get here, we hit max iterations
        raise TaskTimeout(
            f"Task '{task_name}' exceeded maximum iterations ({self.max_iterations})"
        )

    def _run_task_loop(
        self,
        task_name: str,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
        state: Versioned | Namespaced | None,
        on_event: Callable[[Any], None] | None = None,
        setup: str | None = None,
    ):
        """
        Execute the agent task loop.
        This now consumes the generator to provide identical behavior to the streaming version.

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
        generator = self._task_loop_generator(
            task_name,
            docstring,
            inputs_dataclass,
            inputs_instance,
            return_type,
            state,
            on_event=on_event,
            setup=setup,
        )

        try:
            # Consume all events until completion
            while True:
                next(generator)
        except StopIteration as e:
            return e.value  # Generator's return value
        except (TaskFail, TaskClarify):
            raise  # Let TaskFail and TaskClarify propagate normally

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
        from agex.agent.task_messages import build_task_message

        return build_task_message(
            docstring, inputs_dataclass, inputs_instance, return_type
        )

    def _get_llm_response(self, messages, exec_state, on_event):
        """Get structured response with retry; emit ErrorEvent per attempt."""
        import time

        from agex.agent.events import ErrorEvent
        from agex.llm.core import ResponseParseError

        max_retries = max(0, self.llm_max_retries)
        backoff = max(0.0, self.llm_retry_backoff)
        provider = self.llm_client.provider_name
        model = self.llm_client.model

        attempt = 0
        while True:
            try:
                return self.llm_client.complete(messages)
            except (ResponseParseError, RuntimeError) as e:
                is_last = attempt >= max_retries
                # Emit recoverable for retries, fatal for last
                err = ErrorEvent(
                    agent_name=self.name,
                    error=e,
                    recoverable=not is_last,
                )
                from agex.state.log import add_event_to_log

                add_event_to_log(exec_state, err, on_event=on_event)
                if is_last:
                    raise LLMFail(
                        message=str(e), provider=provider, model=model, retries=attempt
                    )
                # Backoff and retry
                sleep_secs = backoff * (2**attempt)
                time.sleep(sleep_secs)
                attempt += 1
