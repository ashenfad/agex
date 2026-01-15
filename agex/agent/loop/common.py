"""
Common helpers, constants, and event factories for the task loop.

This module contains shared logic used by both sync and async task loop implementations.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

from pydantic import ValidationError

from agex.agent.datatypes import (
    LLMFail,
    TaskCancelled,
    TaskClarify,
    TaskContinue,
    TaskFail,
    TaskSuccess,
    TaskTimeout,
    _AgentExit,
)
from agex.agent.events import (
    ActionEvent,
    BaseEvent,
    ClarifyEvent,
    ErrorEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    SystemNoteEvent,
    TaskStartEvent,
)
from agex.eval.error import EvalError
from agex.eval.objects import PrintAction
from agex.llm.core import LLMResponse, ResponseBuilder, ResponseParseError, StreamToken
from agex.state import (
    ConcurrencyError,
    Live,
    Namespaced,
    Versioned,
    events,
    get_commit_hash,
    is_live_root,
)
from agex.state.log import add_event_to_log, get_events_from_log

# Re-export commonly used items for convenience
__all__ = [
    # Constants
    "MAX_USER_FUNCTIONS_IN_RECAP",
    "TASK_CONTROL_GUIDANCE",
    # Event factories
    "create_task_start_event",
    "create_action_event",
    "create_success_event",
    "create_clarify_event",
    "create_fail_event",
    "create_error_output",
    "create_guidance_output",
    "create_unsaved_warning",
    # State helpers
    "get_commit_hash",
    "initialize_exec_state",
    "check_for_task_call",
    "strip_namespace_prefix",
    "yield_new_events",
    "maybe_file_event",
    "maybe_add_file_event",
    "apply_optimistic_file_writes",
    "ResponseBuilder",
    # Re-exports
    "ValidationError",
    "LLMFail",
    "TaskClarify",
    "TaskContinue",
    "TaskFail",
    "TaskSuccess",
    "TaskTimeout",
    "TaskCancelled",
    "_AgentExit",
    "ActionEvent",
    "ClarifyEvent",
    "ErrorEvent",
    "FailEvent",
    "OutputEvent",
    "SuccessEvent",
    "SystemNoteEvent",
    "TaskStartEvent",
    "EvalError",
    "PrintAction",
    "LLMResponse",
    "ResponseParseError",
    "StreamToken",
    "ConcurrencyError",
    "Live",
    "Namespaced",
    "Versioned",
    "events",
    "is_live_root",
    "add_event_to_log",
    "get_events_from_log",
    "check_cancellation",
]

MAX_USER_FUNCTIONS_IN_RECAP = 12


def check_cancellation(
    task_name: str,
    versioned_state: Versioned | None,
    exec_state: Any,
) -> bool:
    """
    Check if a cancellation sentinel is present for the given task.

    Reads directly from the underlying KV store for Versioned state to ensure
    immediate visibility of cancellation requests from other threads/processes.

    Args:
        task_name: Name of the task to check cancellation for
        versioned_state: The Versioned state if present, or None
        exec_state: The execution state (Live or Namespaced)

    Returns:
        True if cancellation was detected (and sentinel was cleaned up), False otherwise
    """
    cancel_key = f"__agex_cancel__{task_name}"

    if isinstance(versioned_state, Versioned):
        # Read directly from KV store for immediate visibility
        if versioned_state.get_raw(cancel_key):
            # Clean up the sentinel
            versioned_state.remove_raw(cancel_key)
            return True
    else:
        # Live/Namespaced state - check exec_state directly
        if exec_state.get(cancel_key):
            exec_state.remove(cancel_key)
            return True

    return False


# Task control guidance message (shown when agent forgets to signal completion)
TASK_CONTROL_GUIDANCE = (
    "💡 **Task Control Reminder**: Your code executed successfully, but you need to signal completion.\n\n"
    "**Next steps:**\n"
    "• `task_success(result)` - Complete the task with your final answer\n"
    "• `task_continue(result)` - Observe your work and continue to another REPL iteration\n"
    "• `task_fail(message)` - If you cannot complete the task\n"
    "• `task_clarify(message)` - If you need more information\n\n"
    "Your code ran without errors - now just add the appropriate task control function!"
)


# =============================================================================
# State Helpers
# =============================================================================


def initialize_exec_state(
    agent_name: str,
    state: Versioned | Live | Namespaced | None,
    inputs_instance: Any,
    return_type: type,
    session: str = "default",
) -> tuple[Versioned | Live | Namespaced, Versioned | None]:
    """
    Initialize the execution state based on the provided state argument.

    Args:
        agent_name: Name of the agent
        state: The state to use for execution
        inputs_instance: The task inputs
        return_type: Expected return type
        session: Session identifier for state resolution (inherited by sub-agents)

    Returns:
        A tuple of (exec_state, versioned_state) where versioned_state is the
        state we're responsible for snapshotting (or None if we don't own it
        or if the state is Live/ephemeral).
    """
    versioned_state: Versioned | None = None
    exec_state: Versioned | Live | Namespaced

    if isinstance(state, Namespaced):
        # Namespaced = someone else owns versioning, we just work within namespace
        exec_state = state
        versioned_state = None
    elif isinstance(state, Versioned):
        # Versioned = we're responsible for versioning this state
        versioned_state = state
        exec_state = state  # No namespacing - use directly
    elif isinstance(state, Live):
        # Live = ephemeral in-memory state, no snapshotting needed
        exec_state = state  # No namespacing - use directly
        versioned_state = None
    else:
        # None = we create and own new live state (no persistence by default)
        exec_state = Live()

    # Add inputs and expected return type to state for agent access
    if inputs_instance is not None:
        exec_state.set("inputs", inputs_instance)
    exec_state.set("__expected_return_type__", return_type)

    # Initialize the event log if it doesn't exist
    if "__event_log__" not in exec_state:
        exec_state.set("__event_log__", [])

    return exec_state, versioned_state


def strip_namespace_prefix(keys: list[str], namespace_prefix: str) -> list[str]:
    """Strip namespace prefix from keys for user-facing messages."""
    result = []
    for key in keys:
        if key.startswith(namespace_prefix):
            result.append(key[len(namespace_prefix) :])
        else:
            result.append(key)
    return result


def check_for_task_call(code: str) -> bool:
    """Check if code contains any task_* function calls."""
    if not code or not code.strip():
        return False
    return any(
        task_func in code
        for task_func in [
            "task_success(",
            "task_fail(",
            "task_clarify(",
            "task_continue(",
        ]
    )


def yield_new_events(
    exec_state, events_yielded_count: int, on_event: Callable | None = None
):
    """
    Generator that yields new events since events_yielded_count.

    Returns the events to yield. Caller is responsible for updating their counter
    to len(events(exec_state)) after consuming.
    """
    all_events = events(exec_state)
    return all_events[events_yielded_count:]


def apply_optimistic_file_writes(
    agent: Any,
    llm_response: LLMResponse,
    fs: Any,
    exec_state: Any,
    on_event: Callable[[BaseEvent], None] | None = None,
) -> None:
    """
    Apply file writes from the LLM response to the filesystem.

    This is called 'optimistic' because it happens before code execution.
    It allows the agent to import modules it just created.
    """
    if not llm_response.files or not fs:
        return

    from agex.fs.aware import AgentAwareFS
    from agex.fs.virtual import VirtualFS

    # Use underlying FS directly to avoid 'user' source attribution
    # and handle snapshot parameter for VirtualFS
    target_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs
    for path, content in llm_response.files.items():
        # Check for shadowing
        if path.endswith(".py"):
            module_name = path[:-3].replace("/", ".")
            if module_name in agent._policy.namespaces:
                warning = SystemNoteEvent(
                    agent_name="System",
                    message=(
                        f"⚠️ Warning: Created file '{path}' shadows registered system "
                        f"module '{module_name}'. The system module will take precedence "
                        f"during imports."
                    ),
                )
                add_event_to_log(exec_state, warning, on_event=on_event)

        if isinstance(target_fs, VirtualFS):
            target_fs.write(path, content.encode("utf-8"), snapshot=False)
        else:
            target_fs.write(path, content.encode("utf-8"))


# =============================================================================
# Event Factories
# =============================================================================


def create_task_start_event(
    agent_name: str,
    task_name: str,
    inputs_dataclass: type,
    inputs_instance: Any,
    message: str,
) -> TaskStartEvent:
    """Create a TaskStartEvent with deep-copied inputs."""
    return TaskStartEvent(
        agent_name=agent_name,
        task_name=task_name,
        inputs={
            f.name: deepcopy(getattr(inputs_instance, f.name))
            for f in inputs_dataclass.__dataclass_fields__.values()
        },
        message=message,
    )


def create_action_event(
    agent_name: str,
    llm_response: LLMResponse,
    source: str = "main",
) -> ActionEvent:
    """Create an ActionEvent from an LLM response."""
    return ActionEvent(
        agent_name=agent_name,
        title=llm_response.title,
        thinking=llm_response.thinking,
        code=llm_response.code,
        files=llm_response.files,
        source=source,
    )


def create_success_event(agent_name: str, result: Any) -> SuccessEvent:
    """Create a SuccessEvent."""
    return SuccessEvent(agent_name=agent_name, result=result)


def create_clarify_event(agent_name: str, message: str) -> ClarifyEvent:
    """Create a ClarifyEvent."""
    return ClarifyEvent(agent_name=agent_name, message=message)


def create_fail_event(agent_name: str, message: str) -> FailEvent:
    """Create a FailEvent."""
    return FailEvent(agent_name=agent_name, message=message)


def create_error_output(agent_name: str, exception: Exception) -> OutputEvent:
    """Create an OutputEvent for an evaluation error."""
    return OutputEvent(
        agent_name=agent_name,
        parts=[
            PrintAction(
                [
                    f"💥 Evaluation error: {exception}\nYou must adjust your code accordingly!"
                ]
            )
        ],
    )


def create_guidance_output(agent_name: str) -> OutputEvent:
    """Create an OutputEvent with task control guidance."""
    return OutputEvent(
        agent_name=agent_name,
        parts=[PrintAction([TASK_CONTROL_GUIDANCE])],
    )


def create_unsaved_warning(
    agent_name: str,
    unsaved_keys: list[str],
    namespace_prefix: str,
) -> OutputEvent:
    """Create an OutputEvent warning about unsaved variables."""
    agent_visible_keys = strip_namespace_prefix(unsaved_keys, namespace_prefix)
    warning_message = (
        f"⚠️ Could not save the following variables because they "
        f"are not serializable: {', '.join(agent_visible_keys)}"
    )
    return OutputEvent(
        agent_name=agent_name,
        parts=[PrintAction([warning_message])],
    )


def create_transient_event(
    message: str, last_timestamp: datetime | None = None
) -> SystemNoteEvent:
    """Create a transient SystemNoteEvent for LLM context."""
    event = SystemNoteEvent(
        agent_name="System",
        message=message,
    )
    if last_timestamp:
        event.timestamp = last_timestamp
    return event


def maybe_file_event(
    agent_name: str,
    metadata_before: dict,
    metadata_after: dict,
) -> None:
    """Emit FileEvent if files changed during agent code execution."""
    from agex.agent.events import FileEvent

    before_paths = set(metadata_before.keys())
    after_paths = set(metadata_after.keys())

    added = list(after_paths - before_paths)
    removed = list(before_paths - after_paths)

    # Modified = same path but different modified_at timestamp
    modified = [
        p
        for p in before_paths & after_paths
        if metadata_before[p].modified_at != metadata_after[p].modified_at
    ]

    # Only emit if something actually changed
    if added or modified or removed:
        return FileEvent(
            agent_name=agent_name,
            file_source="agent",
            added=added,
            modified=modified,
            removed=removed,
        )
    return None


def maybe_add_file_event(
    fs,
    fs_metadata_before: dict,
    exec_state,
    agent_name: str,
    commit_hash: str | None,
):
    """Check for file changes and add FileEvent to log if needed.

    This should be called BEFORE snapshot so the FileEvent is included in the commit.
    The FileEvent should be yielded to the caller, but NOT emitted via on_event yet.
    Emission happens after merge.

    Args:
        fs: The filesystem instance (or None if no fs)
        fs_metadata_before: Snapshot of file metadata before execution
        exec_state: The execution state to add the event to
        agent_name: Name of the agent
        commit_hash: Pre-generated commit hash for the event

    Returns:
        The FileEvent if created, None otherwise
    """

    if not fs:
        return None

    fs_metadata_after = fs.get_metadata_snapshot()
    file_event = maybe_file_event(agent_name, fs_metadata_before, fs_metadata_after)

    if file_event:
        file_event.commit_hash = commit_hash
        # Add to log WITHOUT on_event - we'll emit after merge
        add_event_to_log(exec_state, file_event)

    return file_event
