"""
Synchronous task loop implementation.

Contains the sync versions of the task loop generator and run methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agex.fs.base import FileSystem

from agex.agent.events import CancelledEvent
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
    TaskCancelled,
    TaskClarify,
    TaskContinue,
    TaskFail,
    # Re-exports
    TaskSuccess,
    TaskTimeout,
    Versioned,
    _AgentExit,
    add_event_to_log,
    apply_optimistic_file_writes,
    # Helpers
    check_cancellation,
    check_for_task_call,
    create_action_event,
    create_clarify_event,
    create_error_output,
    create_fail_event,
    create_guidance_output,
    create_success_event,
    # Event factories
    create_task_start_event,
    events,
    # State helpers
    get_commit_hash,
    get_events_from_log,
    initialize_exec_state,
    maybe_add_file_event,
    safe_snapshot,
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
        fs: FileSystem | None,
        fs_metadata_before: dict,
        session: str = "default",
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
            self.name, state, inputs_instance, return_type, session=session
        )
        events_yielded = len(events(exec_state))

        # Clear any stale cancellation signal from a previous run.
        # This handles the race condition where a cancel arrives just as the previous
        # task finishes - we don't want it to immediately cancel this fresh task.
        cancel_key = f"__agex_cancel__{task_name}"
        if versioned_state is not None:
            versioned_state.remove_raw(cancel_key)
        elif hasattr(exec_state, "remove"):
            exec_state.remove(cancel_key)

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
                    fs=fs,
                    session=session,
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
            # Check for cancellation at the start of each iteration
            if check_cancellation(task_name, versioned_state, exec_state):
                # Pre-generate commit hash so the terminal event can reference
                # the commit that will include it
                next_commit = get_commit_hash() if versioned_state else None

                # Record CancelledEvent in the log FIRST
                cancelled_event = CancelledEvent(
                    agent_name=self.name,
                    task_name=task_name,
                    iterations_completed=iteration,
                )
                cancelled_event.commit_hash = next_commit
                add_event_to_log(exec_state, cancelled_event, on_event=on_event)
                yield cancelled_event

                # Snapshot AFTER adding the event so it's included
                if versioned_state is not None:
                    safe_snapshot(versioned_state, commit_hash=next_commit)

                raise TaskCancelled(
                    message=f"Task '{task_name}' was cancelled",
                    task_name=task_name,
                    iterations_completed=iteration,
                )

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
                apply_optimistic_file_writes(
                    self, llm_response, fs, exec_state, on_event=on_event
                )

                if code_to_evaluate:
                    evaluate_program(
                        code_to_evaluate,
                        self,
                        exec_state,
                        self.eval_timeout_seconds,
                        fs=fs,
                        session=session,
                        on_event=on_event,
                        on_token=on_token,
                    )

            except TaskSuccess as task_signal:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                # Pre-generate commit hash so the terminal event can reference
                # the commit that will include it
                next_commit = get_commit_hash() if versioned_state else None

                # Check for file changes and add to log before snapshot
                file_event = maybe_add_file_event(
                    fs, fs_metadata_before, exec_state, self.name, next_commit
                )
                if file_event:
                    yield file_event

                success_event = create_success_event(self.name, task_signal.result)
                success_event.commit_hash = next_commit
                add_event_to_log(exec_state, success_event, on_event=on_event)
                yield success_event

                # Snapshot with the pre-generated hash so event.commit_hash matches
                if versioned_state is not None:
                    safe_snapshot(versioned_state, commit_hash=next_commit)

                return task_signal.result

            except TaskContinue:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                # Persist changes from this iteration (including <file> writes)
                if versioned_state is not None:
                    safe_snapshot(versioned_state)

                continue

            except TaskClarify as task_clarify:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                # Pre-generate commit hash so the terminal event can reference
                # the commit that will include it
                next_commit = get_commit_hash() if versioned_state else None

                # Check for file changes and add to log before snapshot
                file_event = maybe_add_file_event(
                    fs, fs_metadata_before, exec_state, self.name, next_commit
                )
                if file_event:
                    yield file_event

                clarify_event = create_clarify_event(self.name, task_clarify.message)
                clarify_event.commit_hash = next_commit
                add_event_to_log(exec_state, clarify_event, on_event=on_event)
                yield clarify_event

                # Snapshot with the pre-generated hash so event.commit_hash matches
                if versioned_state is not None:
                    safe_snapshot(versioned_state, commit_hash=next_commit)

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

                # Pre-generate commit hash so the terminal event can reference
                # the commit that will include it
                next_commit = get_commit_hash() if versioned_state else None

                # Check for file changes and add to log before snapshot
                file_event = maybe_add_file_event(
                    fs, fs_metadata_before, exec_state, self.name, next_commit
                )
                if file_event:
                    yield file_event

                fail_event = create_fail_event(self.name, task_fail.message)
                fail_event.commit_hash = next_commit
                add_event_to_log(exec_state, fail_event, on_event=on_event)
                yield fail_event

                # Snapshot with the pre-generated hash so event.commit_hash matches
                if versioned_state is not None:
                    safe_snapshot(versioned_state, commit_hash=next_commit)

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
        session: str = "default",
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

        if self._fs_config:
            fs = self.fs(session=session)
            fs_metadata_before = fs.get_metadata_snapshot()
        else:
            fs = None
            fs_metadata_before = {}

        for attempt in range(max_conflict_retries + 1):
            try:
                file_events = []  # Track FileEvents for post-merge emission
                generator = self._task_loop_generator(
                    task_name,
                    docstring,
                    inputs_dataclass,
                    inputs_instance,
                    return_type,
                    state,
                    fs,
                    fs_metadata_before,
                    session=session,
                    on_event=on_event,
                    on_token=on_token,
                    setup=setup,
                )

                try:
                    while True:
                        event = next(generator)
                        # Track FileEvents but don't emit yet - wait for merge
                        from agex.agent.events import FileEvent

                        if isinstance(event, FileEvent):
                            file_events.append(event)
                except StopIteration as e:
                    result = e.value

                if versioned_state is not None:
                    success = versioned_state.merge(on_conflict=on_conflict)
                    if not success:
                        raise ConcurrencyError("Failed to merge state")

                # Emit FileEvents after merge
                for file_event in file_events:
                    if on_event:
                        on_event(file_event)

                return result

            except ConcurrencyError:
                if on_conflict == "abandon":
                    return None
                if attempt >= max_conflict_retries:
                    raise
                if versioned_state is not None:
                    versioned_state.reset()

            except (TaskFail, TaskClarify, _AgentExit):
                if versioned_state is not None:
                    try:
                        if on_conflict == "abandon":
                            versioned_state.merge(on_conflict="abandon")
                        else:
                            versioned_state.merge()

                    except ConcurrencyError:
                        if on_conflict != "abandon":
                            raise

                # Emit FileEvents after merge
                for file_event in file_events:
                    if on_event:
                        on_event(file_event)

                raise
