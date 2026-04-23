"""
Synchronous task loop implementation.

Contains the sync versions of the task loop generator and run methods.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from monkeyfs import FileSystem

    from agex.state.live import Live

from kvgit import Staged

from agex.agent.chapter import CHAPTER_TASK
from agex.agent.emissions import (
    FileEditEmission,
    FileWriteEmission,
    PythonEmission,
    TerminalEmission,
    TextEmission,
    ThinkingEmission,
)
from agex.agent.events import CancelledEvent, SystemNoteEvent
from agex.eval.bridge import execute_sandboxed
from agex.resource_limits import apply_resource_limits
from agex.state import safe_commit

from .common import (
    ActionEvent,
    ConcurrencyError,
    EvalError,
    LLMFail,
    MergeConflict,
    Namespaced,
    TaskCancelled,
    TaskClarify,
    TaskFail,
    TaskSuccess,
    TaskTimeout,
    _AgentExit,
    add_event_to_log,
    apply_file_edit,
    apply_file_write,
    check_cancellation,
    check_for_task_call,
    create_action_event,
    create_clarify_event,
    create_error_output,
    create_fail_event,
    create_guidance_output,
    create_success_event,
    create_task_start_event,
    events,
    execute_terminal,
    get_events_from_log,
    initialize_exec_state,
    maybe_add_file_event,
    yield_new_events,
)
from .state_helpers import (
    clear_stale_cancel,
    collect_python_refs,
    mount_chapters_overlay,
    prepare_task_loop,
    strip_python_fences,
)


def _run_coro(coro):
    """Run a coroutine from sync code, handling nested event loops.

    When no event loop is running, uses ``asyncio.run()``. When already
    inside an async loop (e.g. a sync sub-agent called from an async
    orchestrator), skips execution to avoid ``asyncio.run()`` errors.
    """
    try:
        asyncio.get_running_loop()
        # Already inside an event loop — can't use asyncio.run().
        # Close the coroutine to avoid "was never awaited" warnings.
        coro.close()
        return
    except RuntimeError:
        pass
    asyncio.run(coro)


def _last_python_code(emissions: list) -> str:
    """Return the concatenation of all PythonEmission code bodies.

    Used by ``check_for_task_call`` to decide whether a turn produced
    an explicit terminator or implicitly continues.
    """
    return "\n".join(
        em.code for em in emissions if isinstance(em, PythonEmission) and em.code
    )


def _emission_block_id(event_idx: int, emission_idx: int) -> str:
    """Stable block id derived from position in the event log.

    The renderer computes the same id from the unfiltered event index
    so PrintAction / ImageAction parts emitted here pair cleanly to
    their tool_use blocks at render time.
    """
    return f"em_{event_idx}_{emission_idx}"


class SyncLoopMixin:
    """Mixin providing synchronous task loop methods."""

    def _handle_terminal_condition(
        self,
        exec_state,
        versioned_state,
        fs,
        fs_metadata_before,
        events_yielded,
        terminal_event,
        on_event,
        referenced_keys=None,
    ):
        """Helper to handle common terminal condition logic (success, fail, clarify)."""
        for event in yield_new_events(exec_state, events_yielded):
            yield event

        # Check for file changes and add to log before commit
        file_event = maybe_add_file_event(fs, fs_metadata_before, exec_state, self.name)
        if file_event:
            yield file_event

        add_event_to_log(exec_state, terminal_event, on_event=on_event)
        yield terminal_event

        # Commit with mutation detection
        if versioned_state is not None:
            safe_commit(versioned_state, referenced_keys=referenced_keys)

    def _execute_emissions(
        self,
        action_event: ActionEvent,
        event_idx: int,
        exec_state,
        versioned_state,
        fs,
        session: str,
        on_event,
        on_token,
    ) -> tuple[bool, Exception | None]:
        """Walk an ActionEvent's emissions sequentially.

        Returns ``(ran_terminator, recoverable_error)``.  The caller
        handles the terminator exception (already re-raised out of
        this method) and the recoverable error (logged as an
        OutputEvent, loop continues on next iteration).

        Emissions execute in stream-arrival order.  PythonEmissions
        share state transitively — each call's ``handle_result`` syncs
        assignments back, and the next emission's ``build_namespace``
        hydrates from the updated state.  On the first terminator, the
        walk aborts; remaining emissions stay in the log but don't run.
        """
        recoverable_error: Exception | None = None

        for j, emission in enumerate(action_event.emissions):
            emission_id = _emission_block_id(event_idx, j)

            if isinstance(emission, (TextEmission, ThinkingEmission)):
                continue  # Logged, not executed.

            if isinstance(emission, FileWriteEmission):
                try:
                    apply_file_write(self, emission, fs, exec_state, on_event=on_event)
                    add_event_to_log(
                        exec_state,
                        SystemNoteEvent(
                            agent_name="System",
                            message=f"✓ write_file: {emission.path}",
                        ),
                        on_event=on_event,
                    )
                except Exception as e:
                    recoverable_error = e
                    error_output = create_error_output(
                        self.name, e, emission_id=emission_id
                    )
                    add_event_to_log(exec_state, error_output, on_event=on_event)
                    break

            elif isinstance(emission, FileEditEmission):
                try:
                    apply_file_edit(emission, fs, exec_state, on_event=on_event)
                    add_event_to_log(
                        exec_state,
                        SystemNoteEvent(
                            agent_name="System",
                            message=f"✓ edit_file: {emission.path}",
                        ),
                        on_event=on_event,
                    )
                except Exception as e:
                    recoverable_error = e
                    error_output = create_error_output(
                        self.name, e, emission_id=emission_id
                    )
                    add_event_to_log(exec_state, error_output, on_event=on_event)
                    break

            elif isinstance(emission, PythonEmission):
                code = emission.code or ""
                if not code.strip():
                    continue
                try:
                    with apply_resource_limits(self._resource_limits):
                        execute_sandboxed(
                            code,
                            self,
                            exec_state,
                            self.eval_timeout_seconds,
                            fs=fs,
                            session=session,
                            on_event=on_event,
                            on_token=on_token,
                            emission_id=emission_id,
                        )
                except (TaskSuccess, TaskClarify, TaskFail, LLMFail, _AgentExit):
                    raise
                except Exception as e:
                    recoverable_error = e
                    error_output = create_error_output(
                        self.name, e, emission_id=emission_id
                    )
                    add_event_to_log(exec_state, error_output, on_event=on_event)
                    break

            elif isinstance(emission, TerminalEmission):
                commands = emission.commands or ""
                if not commands.strip():
                    continue
                from agex.agent.loop.event_factories import build_terminal_commands

                terminal_commands = build_terminal_commands(
                    self, fs, state=versioned_state, vfs=fs
                )
                try:
                    with apply_resource_limits(self._resource_limits):
                        execute_terminal(
                            self.name,
                            commands,
                            fs,
                            exec_state,
                            on_event=on_event,
                            commands=terminal_commands or None,
                            emission_id=emission_id,
                        )
                except Exception as e:
                    recoverable_error = e
                    # execute_terminal already logs an OutputEvent for
                    # ParseError / TerminalError; just break.
                    break

        return (False, recoverable_error)

    def _task_loop_generator(
        self,
        task_name: str,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
        state: Staged | Live | Namespaced | None,
        fs: FileSystem | None,
        fs_metadata_before: dict,
        session: str = "default",
        on_event: Callable[[Any], None] | None = None,
        on_token: Callable[[Any], None] | None = None,
        setup: str | None = None,
    ):
        """Generator that yields events as they happen during task execution."""
        # Initialize state
        exec_state, versioned_state = initialize_exec_state(
            self.name, state, inputs_instance, return_type, session=session
        )
        events_yielded = len(events(exec_state))
        clear_stale_cancel(task_name, versioned_state, exec_state)

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

        # Execute setup code if provided — a synthetic "setup" ActionEvent
        # with a single PythonEmission so rendering is consistent with
        # normal turns.
        if setup:
            setup_emission = PythonEmission(
                code=setup,
                title="Setup",
            )
            setup_action_event = ActionEvent(
                agent_name=self.name,
                emissions=[
                    ThinkingEmission(
                        text="This code was automatically run to provide "
                        "context for the task."
                    ),
                    setup_emission,
                ],
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

            # Use the same emission_id formula the renderer will use
            # when this setup event gets re-rendered later.
            setup_event_idx = len(events(exec_state)) - 1
            setup_emission_id = _emission_block_id(setup_event_idx, 1)
            try:
                with apply_resource_limits(self._resource_limits):
                    execute_sandboxed(
                        setup,
                        self,
                        exec_state,
                        self.eval_timeout_seconds,
                        fs=fs,
                        session=session,
                        on_event=setup_on_event,
                        on_token=on_token,
                        emission_id=setup_emission_id,
                    )
            except BaseException:
                pass

            # Yield new events
            for event in yield_new_events(exec_state, events_yielded):
                yield event
            events_yielded = len(events(exec_state))

        # Accumulate referenced state keys across iterations for mutation
        # detection.
        accumulated_refs: set[str] = set()
        last_error: Exception | None = None

        # Main task loop
        for iteration in range(self.max_iterations):
            # Check for cancellation at the start of each iteration
            if check_cancellation(task_name, versioned_state, exec_state):
                cancelled_event = CancelledEvent(
                    agent_name=self.name,
                    task_name=task_name,
                    iterations_completed=iteration,
                )
                add_event_to_log(exec_state, cancelled_event, on_event=on_event)
                yield cancelled_event

                if versioned_state is not None:
                    safe_commit(versioned_state)

                raise TaskCancelled(
                    message=f"Task '{task_name}' was cancelled",
                    task_name=task_name,
                    iterations_completed=iteration,
                )

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
            strip_python_fences(llm_response, self._strip_markdown_code_fence)
            collect_python_refs(llm_response, exec_state, accumulated_refs)

            # Create and yield action event
            action_event = create_action_event(self.name, llm_response)
            add_event_to_log(exec_state, action_event, on_event=on_event)
            yield action_event
            events_yielded += 1
            event_idx = len(events(exec_state)) - 1

            # Walk the emissions.  Terminators (TaskSuccess / TaskFail /
            # TaskClarify) bubble up as exceptions; other exceptions are
            # caught inside _execute_emissions and surfaced as an
            # OutputEvent so the agent can retry on the next iteration.
            try:
                _, recoverable_error = self._execute_emissions(
                    action_event,
                    event_idx,
                    exec_state,
                    versioned_state,
                    fs,
                    session,
                    on_event,
                    on_token,
                )
                if recoverable_error is not None:
                    last_error = recoverable_error

            except TaskSuccess as task_signal:
                success_event = create_success_event(self.name, task_signal.result)
                yield from self._handle_terminal_condition(
                    exec_state,
                    versioned_state,
                    fs,
                    fs_metadata_before,
                    events_yielded,
                    success_event,
                    on_event,
                    referenced_keys=accumulated_refs,
                )
                return task_signal.result

            except TaskClarify as task_clarify:
                clarify_event = create_clarify_event(self.name, task_clarify.message)
                yield from self._handle_terminal_condition(
                    exec_state,
                    versioned_state,
                    fs,
                    fs_metadata_before,
                    events_yielded,
                    clarify_event,
                    on_event,
                    referenced_keys=accumulated_refs,
                )
                if isinstance(state, Namespaced):
                    raise EvalError(
                        f"Sub-agent needs clarification: {task_clarify.message}"
                    )
                else:
                    raise

            except TaskFail as task_fail:
                fail_event = create_fail_event(self.name, task_fail.message)
                yield from self._handle_terminal_condition(
                    exec_state,
                    versioned_state,
                    fs,
                    fs_metadata_before,
                    events_yielded,
                    fail_event,
                    on_event,
                    referenced_keys=accumulated_refs,
                )
                if isinstance(state, Namespaced):
                    raise EvalError(f"Sub-agent failed: {task_fail.message}")
                else:
                    raise

            except LLMFail:
                raise

            except _AgentExit:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))
                raise

            # Yield any events produced during the emission walk.
            for event in yield_new_events(exec_state, events_yielded):
                yield event
            events_yielded = len(events(exec_state))

            # Persist changes from this iteration.
            if versioned_state is not None:
                safe_commit(versioned_state, referenced_keys=accumulated_refs)

            # Nudge if the agent ran out of Python without signaling.
            combined_code = _last_python_code(action_event.emissions)
            if combined_code.strip() and not check_for_task_call(combined_code):
                # Pick the last PythonEmission's id for the nudge so the
                # renderer pairs it with the expected tool_result.
                last_py_idx = None
                for j, em in enumerate(action_event.emissions):
                    if isinstance(em, PythonEmission) and (em.code or "").strip():
                        last_py_idx = j
                nudge_id = (
                    _emission_block_id(event_idx, last_py_idx)
                    if last_py_idx is not None
                    else None
                )
                guidance_output = create_guidance_output(
                    self.name, emission_id=nudge_id
                )
                add_event_to_log(exec_state, guidance_output, on_event=on_event)
                yield guidance_output
                events_yielded += 1

        msg = f"Task '{task_name}' exceeded maximum iterations ({self.max_iterations})"
        if last_error is not None:
            msg += f"\nLast error: {type(last_error).__name__}: {last_error}"
        raise TaskTimeout(msg)

    def _run_task_loop(
        self,
        task_name: str,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
        state: Staged | Namespaced | None,
        session: str = "default",
        on_event: Callable[[Any], None] | None = None,
        on_token: Callable[[Any], None] | None = None,
        setup: str | None = None,
        on_conflict: str = "retry",
        max_conflict_retries: int = 3,
    ):
        """Execute the agent task loop with automatic retry on concurrency conflicts."""
        versioned_state, fs, fs_metadata_before = prepare_task_loop(
            self, state, session
        )

        for attempt in range(max_conflict_retries + 1):
            try:
                file_events = []
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
                        from agex.agent.events import FileEvent

                        if isinstance(event, FileEvent):
                            file_events.append(event)
                except StopIteration as e:
                    result = e.value

                # Emit FileEvents after successful completion
                for file_event in file_events:
                    if on_event:
                        on_event(file_event)

                # Maybe chapter between tasks
                if task_name != CHAPTER_TASK and state is not None:
                    _run_coro(self._maybe_chapter(state, session, on_event, on_token))
                    if fs is not None:
                        mount_chapters_overlay(fs, state)

                return result

            except (ConcurrencyError, MergeConflict):
                if on_conflict == "abandon":
                    return None
                if attempt >= max_conflict_retries:
                    raise
                if versioned_state is not None:
                    versioned_state.refresh()

            except (TaskFail, TaskClarify, _AgentExit):
                for file_event in file_events:
                    if on_event:
                        on_event(file_event)

                raise
