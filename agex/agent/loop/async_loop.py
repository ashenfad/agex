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
    from monkeyfs import FileSystem

    from agex.state.live import Live

from kvgit import Staged

from agex.agent.chapter import CHAPTER_TASK
from agex.agent.emissions import (
    ACTION_EMISSION_TYPES,
    FileEditEmission,
    FileWriteEmission,
    PythonEmission,
    TerminalEmission,
    TextEmission,
    ThinkingEmission,
)
from agex.agent.events import CancelledEvent, SystemNoteEvent
from agex.agent.utils import call_sync_or_async
from agex.eval.bridge import aexecute_sandboxed, execute_sandboxed
from agex.resource_limits import apply_resource_limits
from agex.state import safe_commit

from .common import (
    ActionEvent,
    ConcurrencyError,
    EvalError,
    LLMFail,
    MergeConflict,
    Namespaced,
    SuccessEvent,
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
    check_for_terminator_call,
    create_action_event,
    create_clarify_event,
    create_error_output,
    create_fail_event,
    create_guidance_output,
    create_no_progress_guidance,
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
from .sync_loop import _emission_block_id, _last_python_code


def _execute_with_limits(limits, code, agent, exec_state, timeout, **kwargs):
    """Run execute_sandboxed within resource limits (sync, for executor)."""
    with apply_resource_limits(limits):
        return execute_sandboxed(code, agent, exec_state, timeout, **kwargs)


async def _aexecute_with_limits(limits, code, agent, exec_state, timeout, **kwargs):
    """Run aexecute_sandboxed within resource limits."""
    with apply_resource_limits(limits):
        return await aexecute_sandboxed(code, agent, exec_state, timeout, **kwargs)


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

        file_event = maybe_add_file_event(fs, fs_metadata_before, exec_state, self.name)
        if file_event:
            yield file_event

        add_event_to_log(exec_state, terminal_event, on_event=None)
        if on_event:
            res = call_sync_or_async(on_event, terminal_event)
            if inspect.isawaitable(res):
                await res
        yield terminal_event

        if versioned_state is not None:
            safe_commit(versioned_state, referenced_keys=referenced_keys)

    async def _aexecute_emissions(
        self,
        action_event: ActionEvent,
        event_idx: int,
        exec_state,
        versioned_state,
        fs,
        session: str,
        on_event,
        on_token,
        thread_safe_on_event,
    ) -> tuple[bool, Exception | None]:
        """Async counterpart to :meth:`SyncLoopMixin._execute_emissions`."""
        loop = asyncio.get_running_loop()
        recoverable_error: Exception | None = None

        for j, emission in enumerate(action_event.emissions):
            emission_id = _emission_block_id(event_idx, j)

            if isinstance(emission, (TextEmission, ThinkingEmission)):
                continue

            if isinstance(emission, FileWriteEmission):
                try:
                    with apply_resource_limits(self._resource_limits):
                        apply_file_write(
                            self, emission, fs, exec_state, on_event=on_event
                        )
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
                    add_event_to_log(exec_state, error_output, on_event=None)
                    if on_event:
                        res = call_sync_or_async(on_event, error_output)
                        if inspect.isawaitable(res):
                            await res
                    break

            elif isinstance(emission, FileEditEmission):
                try:
                    with apply_resource_limits(self._resource_limits):
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
                    add_event_to_log(exec_state, error_output, on_event=None)
                    if on_event:
                        res = call_sync_or_async(on_event, error_output)
                        if inspect.isawaitable(res):
                            await res
                    break

            elif isinstance(emission, PythonEmission):
                code = emission.code or ""
                if not code.strip():
                    continue
                try:
                    await _aexecute_with_limits(
                        self._resource_limits,
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
                    add_event_to_log(exec_state, error_output, on_event=None)
                    if on_event:
                        res = call_sync_or_async(on_event, error_output)
                        if inspect.isawaitable(res):
                            await res
                    break

            elif isinstance(emission, TerminalEmission):
                commands = emission.commands or ""
                if not commands.strip():
                    continue
                from agex.agent.loop.event_factories import build_terminal_commands

                terminal_commands = build_terminal_commands(
                    self, fs, state=versioned_state, vfs=fs
                )
                ctx = contextvars.copy_context()
                try:
                    with apply_resource_limits(self._resource_limits):
                        await loop.run_in_executor(
                            None,
                            partial(
                                ctx.run,
                                execute_terminal,
                                self.name,
                                commands,
                                fs,
                                exec_state,
                                thread_safe_on_event,
                                terminal_commands or None,
                                emission_id,
                            ),
                        )
                except Exception as e:
                    recoverable_error = e
                    break

        return (False, recoverable_error)

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
        """Async version of :meth:`SyncLoopMixin._task_loop_generator`."""
        loop = asyncio.get_running_loop()

        exec_state, versioned_state = initialize_exec_state(
            self.name, state, inputs_instance, return_type, session=session
        )
        events_yielded = len(events(exec_state))
        clear_stale_cancel(task_name, versioned_state, exec_state)

        system_message = self._build_system_message()
        initial_task_message = self._build_task_message(
            docstring, inputs_dataclass, inputs_instance, return_type
        )

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

        if setup:
            setup_action_event = ActionEvent(
                agent_name=self.name,
                emissions=[
                    ThinkingEmission(
                        text="This code was automatically run to provide "
                        "context for the task."
                    ),
                    PythonEmission(code=setup, title="Setup"),
                ],
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

            setup_event_idx = len(events(exec_state)) - 1
            setup_emission_id = _emission_block_id(setup_event_idx, 1)
            try:
                await _aexecute_with_limits(
                    self._resource_limits,
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

            for event in yield_new_events(exec_state, events_yielded):
                yield event
            events_yielded = len(events(exec_state))

        accumulated_refs: set[str] = set()
        last_error: Exception | None = None

        for iteration in range(self.max_iterations):
            if check_cancellation(task_name, versioned_state, exec_state):
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

                if versioned_state is not None:
                    safe_commit(versioned_state)

                raise TaskCancelled(
                    message=f"Task '{task_name}' was cancelled",
                    task_name=task_name,
                    iterations_completed=iteration,
                )

            all_events = get_events_from_log(exec_state)

            forefront_msg = self._get_forefront_message(iteration, exec_state)

            llm_response = await self._aget_llm_response(
                system_message,
                all_events,
                exec_state,
                on_event,
                on_token,
                transient_message=forefront_msg,
            )
            strip_python_fences(llm_response, self._strip_markdown_code_fence)
            collect_python_refs(llm_response, exec_state, accumulated_refs)

            action_event = create_action_event(self.name, llm_response)
            add_event_to_log(exec_state, action_event, on_event=None)
            if on_event:
                res = call_sync_or_async(on_event, action_event)
                if inspect.isawaitable(res):
                    await res
            yield action_event
            events_yielded += 1
            event_idx = len(events(exec_state)) - 1

            try:
                _, recoverable_error = await self._aexecute_emissions(
                    action_event,
                    event_idx,
                    exec_state,
                    versioned_state,
                    fs,
                    session,
                    on_event,
                    on_token,
                    thread_safe_on_event,
                )
                if recoverable_error is not None:
                    last_error = recoverable_error

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
                        f"Sub-agent needs clarification: {task_clarify.message}"
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

            for event in yield_new_events(exec_state, events_yielded):
                yield event
            events_yielded = len(events(exec_state))

            if versioned_state is not None:
                safe_commit(versioned_state, referenced_keys=accumulated_refs)

            combined_code = _last_python_code(action_event.emissions)
            if combined_code.strip() and not check_for_terminator_call(combined_code):
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
                add_event_to_log(exec_state, guidance_output, on_event=None)
                if on_event:
                    res = call_sync_or_async(on_event, guidance_output)
                    if inspect.isawaitable(res):
                        await res
                yield guidance_output
                events_yielded += 1
            elif not any(
                isinstance(em, ACTION_EMISSION_TYPES) for em in action_event.emissions
            ):
                no_progress = create_no_progress_guidance(self.name)
                add_event_to_log(exec_state, no_progress, on_event=None)
                if on_event:
                    res = call_sync_or_async(on_event, no_progress)
                    if inspect.isawaitable(res):
                        await res
                yield no_progress
                events_yielded += 1

        msg = f"Task '{task_name}' exceeded maximum iterations ({self.max_iterations})"
        if last_error is not None:
            msg += f"\nLast error: {type(last_error).__name__}: {last_error}"
        raise TaskTimeout(msg)

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

        versioned_state, fs, fs_metadata_before = prepare_task_loop(
            self, state, session
        )

        for attempt in range(max_conflict_retries + 1):
            try:
                result = None
                file_events = []
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
                    from agex.agent.events import FileEvent

                    if isinstance(event, FileEvent):
                        file_events.append(event)
                    elif isinstance(event, SuccessEvent):
                        result = event.result

                for file_event in file_events:
                    if on_event:
                        on_event(file_event)

                if task_name != CHAPTER_TASK and state is not None:
                    await self._maybe_chapter(
                        state,
                        session,
                        on_event,
                        on_token,
                    )
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
