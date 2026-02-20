"""
Asynchronous task loop implementation.

Contains the async versions of the task loop generator and run methods.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agex.fs.base import FileSystem

from kvit import Staged

from agex.agent.events import CancelledEvent
from agex.agent.summarization import maybe_summarize_event_log
from agex.agent.utils import call_sync_or_async
from agex.eval.core import evaluate_program
from agex.resource_limits import apply_resource_limits
from agex.state import get_root, raw_remove, safe_commit

from .common import (
    # Constants
    ActionEvent,
    ConcurrencyError,
    EvalError,
    Live,
    LLMFail,
    MergeConflict,
    Namespaced,
    SuccessEvent,
    TaskCancelled,
    TaskClarify,
    TaskContinue,
    TaskFail,
    # Re-exports
    TaskSuccess,
    TaskTimeout,
    _AgentExit,
    add_event_to_log,
    apply_optimistic_file_actions,
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
    # Terminal execution
    execute_terminal,
    get_events_from_log,
    initialize_exec_state,
    maybe_add_file_event,
    yield_new_events,
)


def _evaluate_with_limits(limits, code, agent, exec_state, timeout, **kwargs):
    """Run evaluate_program within resource limits."""
    with apply_resource_limits(limits):
        return evaluate_program(code, agent, exec_state, timeout, **kwargs)


class AsyncLoopMixin:
    """Mixin providing asynchronous task loop methods."""

    async def _ahandle_terminal_condition(
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

        add_event_to_log(exec_state, terminal_event, on_event=None)
        if on_event:
            res = call_sync_or_async(on_event, terminal_event)
            if inspect.isawaitable(res):
                await res
        yield terminal_event

        # Commit with mutation detection
        if versioned_state is not None:
            safe_commit(versioned_state, referenced_keys=referenced_keys)

    async def _atask_loop_generator(
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
        """
        Async version of _task_loop_generator.
        """
        loop = asyncio.get_running_loop()

        # Initialize state
        exec_state, versioned_state = initialize_exec_state(
            self.name, state, inputs_instance, return_type, session=session
        )
        events_yielded = len(events(exec_state))

        # Clear any stale cancellation signal from a previous run.
        # This handles the race condition where a cancel arrives just as the previous
        # task finishes - we don't want it to immediately cancel this fresh task.
        cancel_key = f"__agex_cancel__{task_name}"
        if isinstance(versioned_state, Staged):
            raw_remove(versioned_state, cancel_key)
        elif hasattr(exec_state, "remove"):
            exec_state.remove(cancel_key)

        # Build messages
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
        add_event_to_log(exec_state, task_start_event, on_event=None)
        if on_event:
            res = call_sync_or_async(on_event, task_start_event)
            if inspect.isawaitable(res):
                await res
        yield task_start_event
        events_yielded += 1

        # Thread-safe wrappers for executor callbacks
        def thread_safe_on_event(event):
            if on_event:
                res = call_sync_or_async(on_event, event, loop=loop)
                if hasattr(res, "result"):
                    try:
                        res.result()
                    except Exception:
                        pass

        def thread_safe_on_token(token):
            if on_token:
                res = call_sync_or_async(on_token, token, loop=loop)
                if hasattr(res, "result"):
                    try:
                        res.result()
                    except Exception:
                        pass

        # Execute setup code if provided
        if setup:
            setup_action_event = ActionEvent(
                agent_name=self.name,
                thinking="This code was automatically run to provide context for the task.",
                code=setup,
                source="setup",
            )
            add_event_to_log(exec_state, setup_action_event, on_event=None)
            if on_event:
                res = call_sync_or_async(on_event, setup_action_event)
                if inspect.isawaitable(res):
                    await res
            yield setup_action_event
            events_yielded += 1

            def setup_on_event(event):
                if event.source == "main":
                    event.source = "setup"
                thread_safe_on_event(event)

            try:
                # Copy context to preserve ContextVars (like agent registry) in thread pool
                ctx = contextvars.copy_context()
                await loop.run_in_executor(
                    None,
                    partial(
                        ctx.run,
                        _evaluate_with_limits,
                        self._resource_limits,
                        setup,
                        self,
                        exec_state,
                        self.eval_timeout_seconds,
                        fs=fs,
                        session=session,
                        on_event=setup_on_event,
                        on_token=thread_safe_on_token,
                        main_loop=loop,
                    ),
                )
            except Exception:
                pass

            for event in yield_new_events(exec_state, events_yielded):
                yield event
            events_yielded = len(events(exec_state))

        # Accumulate referenced state keys across iterations for mutation
        # detection.  find_refs is called once per iteration and the results
        # are unioned so that in-place mutations from earlier iterations are
        # still detected at commit time.
        accumulated_refs: set[str] = set()

        # Main task loop
        for iteration in range(self.max_iterations):
            # Check for cancellation at the start of each iteration
            if check_cancellation(task_name, versioned_state, exec_state):
                # Record CancelledEvent in the log FIRST
                cancelled_event = CancelledEvent(
                    agent_name=self.name,
                    task_name=task_name,
                    iterations_completed=iteration,
                )
                add_event_to_log(exec_state, cancelled_event, on_event=None)
                if on_event:
                    res = call_sync_or_async(on_event, cancelled_event)
                    if inspect.isawaitable(res):
                        await res
                yield cancelled_event

                # Commit AFTER adding the event so it's included
                if versioned_state is not None:
                    safe_commit(versioned_state)

                raise TaskCancelled(
                    message=f"Task '{task_name}' was cancelled",
                    task_name=task_name,
                    iterations_completed=iteration,
                )

            maybe_summarize_event_log(self, exec_state, system_message, on_event)
            all_events = get_events_from_log(exec_state)
            forefront_msg = self._get_forefront_message(iteration, exec_state)

            # Get LLM response (async)
            llm_response = await self._aget_llm_response(
                system_message,
                all_events,
                exec_state,
                on_event,
                on_token,
                transient_message=forefront_msg,
            )
            llm_response.code = self._strip_markdown_code_fence(llm_response.code)
            code_to_evaluate = llm_response.code
            if code_to_evaluate:
                from sblite import find_refs

                try:
                    accumulated_refs |= find_refs(
                        code_to_evaluate, namespace=exec_state
                    )
                except SyntaxError:
                    pass

            # Create and yield action event
            action_event = create_action_event(self.name, llm_response)
            add_event_to_log(exec_state, action_event, on_event=None)
            if on_event:
                res = call_sync_or_async(on_event, action_event)
                if inspect.isawaitable(res):
                    await res
            yield action_event
            events_yielded += 1

            # Evaluate terminal or code
            try:
                # Apply resource limits for file actions (runs in main async context)
                with apply_resource_limits(self._resource_limits):
                    apply_optimistic_file_actions(
                        self, llm_response, fs, exec_state, on_event=on_event
                    )

                if llm_response.terminal:
                    # Execute terminal script in executor - implicitly continues
                    ctx = contextvars.copy_context()
                    await loop.run_in_executor(
                        None,
                        partial(
                            ctx.run,
                            execute_terminal,
                            self.name,
                            llm_response.terminal,
                            fs,
                            exec_state,
                            thread_safe_on_event,
                        ),
                    )

                    # Yield any events from terminal execution
                    for event in yield_new_events(exec_state, events_yielded):
                        yield event
                    events_yielded = len(events(exec_state))

                    # Persist changes from this iteration
                    if versioned_state is not None:
                        safe_commit(versioned_state, referenced_keys=accumulated_refs)

                    continue  # Terminal implicitly continues to next iteration

                elif code_to_evaluate:
                    # Copy context to preserve ContextVars (like agent registry) in thread pool
                    ctx = contextvars.copy_context()
                    await loop.run_in_executor(
                        None,
                        partial(
                            ctx.run,
                            _evaluate_with_limits,
                            self._resource_limits,
                            code_to_evaluate,
                            self,
                            exec_state,
                            self.eval_timeout_seconds,
                            fs=fs,
                            session=session,
                            on_event=thread_safe_on_event,
                            on_token=thread_safe_on_token,
                            main_loop=loop,
                        ),
                    )

            except TaskSuccess as task_signal:
                success_event = create_success_event(self.name, task_signal.result)
                async for event in self._ahandle_terminal_condition(
                    exec_state,
                    versioned_state,
                    fs,
                    fs_metadata_before,
                    events_yielded,
                    success_event,
                    on_event,
                    referenced_keys=accumulated_refs,
                ):
                    yield event
                return

            except TaskContinue:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                # Persist changes from this iteration (including <file> writes)
                if versioned_state is not None:
                    safe_commit(versioned_state, referenced_keys=accumulated_refs)

                continue

            except TaskClarify as task_clarify:
                clarify_event = create_clarify_event(self.name, task_clarify.message)
                async for event in self._ahandle_terminal_condition(
                    exec_state,
                    versioned_state,
                    fs,
                    fs_metadata_before,
                    events_yielded,
                    clarify_event,
                    on_event,
                    referenced_keys=accumulated_refs,
                ):
                    yield event

                if isinstance(state, Namespaced):
                    raise EvalError(
                        f"Sub-agent needs clarification: {task_clarify.message}", None
                    )
                else:
                    raise

            except TaskFail as task_fail:
                fail_event = create_fail_event(self.name, task_fail.message)
                async for event in self._ahandle_terminal_condition(
                    exec_state,
                    versioned_state,
                    fs,
                    fs_metadata_before,
                    events_yielded,
                    fail_event,
                    on_event,
                    referenced_keys=accumulated_refs,
                ):
                    yield event

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
                add_event_to_log(exec_state, error_output, on_event=None)
                if on_event:
                    res = call_sync_or_async(on_event, error_output)
                    if inspect.isawaitable(res):
                        await res
                yield error_output
                events_yielded += 1

            else:
                for event in yield_new_events(exec_state, events_yielded):
                    yield event
                events_yielded = len(events(exec_state))

                if not check_for_task_call(code_to_evaluate):
                    guidance_output = create_guidance_output(self.name)
                    add_event_to_log(exec_state, guidance_output, on_event=None)
                    if on_event:
                        res = call_sync_or_async(on_event, guidance_output)
                        if inspect.isawaitable(res):
                            await res
                    yield guidance_output
                    events_yielded += 1

        raise TaskTimeout(
            f"Task '{task_name}' exceeded maximum iterations ({self.max_iterations})"
        )

    async def _arun_task_loop(
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
        """Async version of _run_task_loop."""

        versioned_state: Staged | None = None
        if isinstance(state, Staged):
            versioned_state = state
        elif isinstance(state, Namespaced):
            base = get_root(state)
            if isinstance(base, Staged):
                versioned_state = base

        if self._fs_config:
            # Use raw VFS for agent execution (not AgentAwareFS which emits per-write events)
            fs, _ = self._get_fs_backend(session)
            fs_metadata_before = fs.get_metadata_snapshot()
        else:
            fs = None
            fs_metadata_before = {}

        for attempt in range(max_conflict_retries + 1):
            try:
                result = None
                file_events = []  # Track FileEvents for post-merge emission
                generator = self._atask_loop_generator(
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

                async for event in generator:
                    # Track FileEvents but don't emit yet - wait for completion
                    from agex.agent.events import FileEvent

                    if isinstance(event, FileEvent):
                        file_events.append(event)
                    elif isinstance(event, SuccessEvent):
                        result = event.result

                # Emit FileEvents after successful completion
                for file_event in file_events:
                    if on_event:
                        on_event(file_event)

                return result

            except (ConcurrencyError, MergeConflict):
                if on_conflict == "abandon":
                    return None
                if attempt >= max_conflict_retries:
                    raise
                if versioned_state is not None:
                    versioned_state.refresh()

            except (TaskFail, TaskClarify, _AgentExit):
                # Emit FileEvents before re-raising
                for file_event in file_events:
                    if on_event:
                        on_event(file_event)

                raise
