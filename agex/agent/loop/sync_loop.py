"""
Synchronous task loop implementation.

Contains the sync versions of the task loop generator and run methods.
"""

from __future__ import annotations

from typing import Any, Callable

from agex.agent.summarization import maybe_summarize_event_log
from agex.eval.core import evaluate_program

from .common import (
    # Constants
    ActionEvent,
    ConcurrencyError,
    EvalError,
    Live,
    LLMFail,
    Namespaced,
    TaskClarify,
    TaskContinue,
    TaskFail,
    # Re-exports
    TaskSuccess,
    TaskTimeout,
    Versioned,
    _AgentExit,
    add_event_to_log,
    check_for_task_call,
    create_action_event,
    create_clarify_event,
    create_error_output,
    create_fail_event,
    create_guidance_output,
    create_success_event,
    # Event factories
    create_task_start_event,
    create_unsaved_warning,
    events,
    get_events_from_log,
    # State helpers
    initialize_exec_state,
    is_live_root,
    yield_new_events,
)


class SyncLoopMixin:
    """Mixin providing synchronous task loop methods."""

    def _task_loop_generator(
        self,
        task_name: str,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
        state: Versioned | Live | Namespaced | None,
        on_event: Callable[[Any], None] | None = None,
        on_token: Callable[[Any], None] | None = None,
        setup: str | None = None,
    ):
        """
        Generator that yields events as they happen during task execution.
        This is the core implementation used by both streaming and regular modes.
        """
        # Initialize state
        exec_state, versioned_state = initialize_exec_state(
            self.name, state, inputs_instance, return_type
        )
        events_yielded = len(events(exec_state))

        # Build system message and initial task message
        system_message = self._build_system_message()
        initial_task_message = self._build_task_message(
            docstring, inputs_dataclass, inputs_instance, return_type
        )

        # Create and yield task start event
        task_start_event = create_task_start_event(
            self.name,
            task_name,
            inputs_dataclass,
            inputs_instance,
            initial_task_message,
        )
        add_event_to_log(exec_state, task_start_event, on_event=on_event)
        yield task_start_event
        events_yielded += 1

        # Execute setup code if provided
        if setup:
            setup_action_event = ActionEvent(
                agent_name=self.name,
                thinking="This code was automatically run to provide context for the task.",
                code=setup,
                source="setup",
            )
            add_event_to_log(exec_state, setup_action_event, on_event=on_event)
            yield setup_action_event
            events_yielded += 1

            def setup_on_event(event):
                if event.source == "main":
                    event.source = "setup"
                if on_event is not None:
                    on_event(event)

            try:
                evaluate_program(
                    setup,
                    self,
                    exec_state,
                    self.eval_timeout_seconds,
                    on_event=setup_on_event,
                    on_token=on_token,
                )
            except Exception:
                pass

            # Yield new events
            for event in yield_new_events(exec_state, events_yielded):
                yield event
            events_yielded = len(events(exec_state))

        # Main task loop
        for iteration in range(self.max_iterations):
            maybe_summarize_event_log(self, exec_state, system_message, on_event)
            all_events = get_events_from_log(exec_state)
            forefront_msg = self._get_forefront_message(iteration, exec_state)

            # Get LLM response
            llm_response = self._get_llm_response(
                system_message,
                all_events,
                exec_state,
                on_event,
                on_token,
                transient_message=forefront_msg,
            )
            llm_response.code = self._strip_markdown_code_fence(llm_response.code)
            code_to_evaluate = llm_response.code

            # Create and yield action event
            action_event = create_action_event(self.name, llm_response)
            add_event_to_log(exec_state, action_event, on_event=on_event)
            yield action_event
            events_yielded += 1

            # Evaluate the code
            try:
                if code_to_evaluate:
                    evaluate_program(
                        code_to_evaluate,
                        self,
                        exec_state,
                        self.eval_timeout_seconds,
                        on_event=on_event,
                        on_token=on_token,
                    )

            except TaskSuccess as task_signal:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                success_event = create_success_event(self.name, task_signal.result)
                add_event_to_log(exec_state, success_event, on_event=on_event)
                yield success_event
                return task_signal.result

            except TaskContinue:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))
                continue

            except TaskClarify as task_clarify:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                clarify_event = create_clarify_event(self.name, task_clarify.message)
                add_event_to_log(exec_state, clarify_event, on_event=on_event)
                yield clarify_event

                if isinstance(state, Namespaced):
                    raise EvalError(
                        f"Sub-agent needs clarification: {task_clarify.message}", None
                    )
                else:
                    raise

            except TaskFail as task_fail:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                fail_event = create_fail_event(self.name, task_fail.message)
                add_event_to_log(exec_state, fail_event, on_event=on_event)
                yield fail_event

                if isinstance(state, Namespaced):
                    raise EvalError(f"Sub-agent failed: {task_fail.message}", None)
                else:
                    raise

            except LLMFail:
                raise

            except _AgentExit:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))
                raise

            except Exception as e:
                error_output = create_error_output(self.name, e)
                add_event_to_log(exec_state, error_output, on_event=on_event)
                yield error_output
                events_yielded += 1

            else:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                if not check_for_task_call(code_to_evaluate):
                    guidance_output = create_guidance_output(self.name)
                    add_event_to_log(exec_state, guidance_output, on_event=on_event)
                    yield guidance_output
                    events_yielded += 1

            finally:
                if versioned_state is not None and not is_live_root(exec_state):
                    result = versioned_state.snapshot()
                    if result.unsaved_keys:
                        warning_output = create_unsaved_warning(
                            self.name, result.unsaved_keys, f"{self.name}/"
                        )
                        add_event_to_log(exec_state, warning_output, on_event=on_event)
                        yield warning_output
                        events_yielded += 1

        # Final snapshot
        if versioned_state is not None:
            result = versioned_state.snapshot()
            if result.unsaved_keys:
                warning_output = create_unsaved_warning(
                    self.name, result.unsaved_keys, f"{self.name}/"
                )
                add_event_to_log(exec_state, warning_output, on_event=on_event)
                yield warning_output
                events_yielded += 1

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
        on_token: Callable[[Any], None] | None = None,
        setup: str | None = None,
        on_conflict: str = "retry",
        max_conflict_retries: int = 3,
    ):
        """
        Execute the agent task loop with automatic retry on concurrency conflicts.
        """
        versioned_state: Versioned | None = None
        if isinstance(state, Versioned):
            versioned_state = state
        elif isinstance(state, Namespaced):
            base = state.base_store
            if isinstance(base, Versioned):
                versioned_state = base

        for attempt in range(max_conflict_retries + 1):
            try:
                generator = self._task_loop_generator(
                    task_name,
                    docstring,
                    inputs_dataclass,
                    inputs_instance,
                    return_type,
                    state,
                    on_event=on_event,
                    on_token=on_token,
                    setup=setup,
                )

                try:
                    while True:
                        next(generator)
                except StopIteration as e:
                    result = e.value

                if versioned_state is not None:
                    if on_conflict == "abandon":
                        versioned_state.merge(on_conflict="abandon")
                    else:
                        versioned_state.merge()

                return result

            except ConcurrencyError:
                if on_conflict == "abandon":
                    return None
                if attempt >= max_conflict_retries:
                    raise
                if versioned_state is not None:
                    versioned_state.reset()

            except (TaskFail, TaskClarify):
                if versioned_state is not None:
                    try:
                        if on_conflict == "abandon":
                            versioned_state.merge(on_conflict="abandon")
                        else:
                            versioned_state.merge()
                    except ConcurrencyError:
                        if on_conflict != "abandon":
                            raise
                raise
