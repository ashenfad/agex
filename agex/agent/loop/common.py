"""
Common helpers, constants, and event factories for the task loop.

This module contains shared logic used by both sync and async task loop implementations.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

from pydantic import ValidationError

from agex.agent.datatypes import (
    EditAction,
    FileAction,
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
from agex.fs.aware import AgentAwareFS
from agex.fs.context import suspend_fs_interception
from agex.fs.virtual import VirtualFS
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
from agex.state.versioned import SnapshotResult


def safe_snapshot(
    versioned_state: Versioned, commit_hash: str | None = None
) -> SnapshotResult:
    """Snapshot state with filesystem interception suspended.

    This ensures that any I/O performed by KV backends during snapshot
    (e.g., disk writes) doesn't get intercepted by VFS/IsolatedFS patching.
    Critical for hierarchical agents where sub-agent snapshots occur while
    parent's filesystem interception is still active.

    Args:
        versioned_state: The versioned state to snapshot.
        commit_hash: Optional pre-generated commit hash.

    Returns:
        SnapshotResult from the underlying snapshot call.
    """
    with suspend_fs_interception():
        return versioned_state.snapshot(commit_hash=commit_hash)


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
    # Terminal execution
    "execute_terminal",
    # State helpers
    "get_commit_hash",
    "initialize_exec_state",
    "check_for_task_call",
    "strip_namespace_prefix",
    "yield_new_events",
    "maybe_file_event",
    "maybe_add_file_event",
    "apply_optimistic_file_actions",
    "safe_snapshot",
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


def _build_trailing_ws_pattern(search: str) -> re.Pattern:
    """Build a regex pattern that matches search with flexible trailing whitespace.

    This allows the search to match even if the file has trailing spaces/tabs
    at the end of lines that the agent didn't include in the search string.
    Internal whitespace (indentation) is preserved exactly.
    """
    lines = search.split("\n")
    pattern_parts = []
    for line in lines:
        # Escape the line for regex, strip trailing whitespace
        escaped = re.escape(line.rstrip())
        # Allow optional trailing whitespace (spaces/tabs only, not newlines)
        pattern_parts.append(escaped + r"[ \t]*")

    # Join with literal newline
    pattern = "\n".join(pattern_parts)
    return re.compile(pattern)


def apply_optimistic_file_actions(
    agent: Any,
    llm_response: LLMResponse,
    fs: Any,
    exec_state: Any,
    on_event: Callable[[BaseEvent], None] | None = None,
) -> None:
    """
    Apply file operations (writes and edits) from the LLM response to the filesystem.

    This is called 'optimistic' because it happens before code execution.
    It allows the agent to import modules it just created.
    """
    if not llm_response.file_actions or not fs:
        return

    # Use underlying FS directly to avoid 'user' source attribution
    # and handle snapshot parameter for VirtualFS
    target_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs

    for action in llm_response.file_actions:
        if isinstance(action, FileAction):
            path, content, mode = action.path, action.content, action.mode
            fs_mode = "a" if mode == "append" else "w"

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
                target_fs.write(
                    path, content.encode("utf-8"), snapshot=False, mode=fs_mode
                )
            else:
                target_fs.write(path, content.encode("utf-8"), mode=fs_mode)

        elif isinstance(action, EditAction):
            path = action.path

            # Read existing file
            try:
                existing_content = target_fs.read(path).decode("utf-8")
            except FileNotFoundError:
                raise ResponseParseError(f"File not found for EDIT: {path}")

            # Try exact match first
            count = existing_content.count(action.search)
            use_normalized = False

            if count == 0:
                # Exact match failed - try normalized matching (flexible trailing whitespace)
                pattern = _build_trailing_ws_pattern(action.search)
                matches = list(pattern.finditer(existing_content))
                count = len(matches)
                use_normalized = True

                if count == 0:
                    search_preview = (
                        action.search[:100] + "..."
                        if len(action.search) > 100
                        else action.search
                    )
                    raise ResponseParseError(
                        f"Search string not found in {path}:\n{search_preview}"
                    )

            if count > 1 and not action.match_all:
                raise ResponseParseError(
                    f"Search string found {count} times in {path}. "
                    f'Use match_all="true" or provide more context.'
                )

            # Apply replacement based on matching mode
            if use_normalized:
                # Normalized matching - use regex replacement
                pattern = _build_trailing_ws_pattern(action.search)

                def make_replacement(match):
                    matched_text = match.group(0)
                    if action.operation == "insert-after":
                        return matched_text + action.content
                    elif action.operation == "insert-before":
                        return action.content + matched_text
                    else:  # "replace"
                        return action.content

                if action.match_all:
                    new_content = pattern.sub(make_replacement, existing_content)
                else:
                    new_content = pattern.sub(
                        make_replacement, existing_content, count=1
                    )
            else:
                # Exact matching - use str.replace
                if action.operation == "insert-after":
                    replacement = action.search + action.content
                elif action.operation == "insert-before":
                    replacement = action.content + action.search
                else:  # "replace"
                    replacement = action.content

                if action.match_all:
                    new_content = existing_content.replace(action.search, replacement)
                else:
                    new_content = existing_content.replace(
                        action.search, replacement, 1
                    )

            # Write back
            if isinstance(target_fs, VirtualFS):
                target_fs.write(
                    path, new_content.encode("utf-8"), snapshot=False, mode="w"
                )
            else:
                target_fs.write(path, new_content.encode("utf-8"), mode="w")


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
        terminal=llm_response.terminal,
        file_actions=llm_response.file_actions,
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


def execute_terminal(
    agent_name: str,
    terminal_script: str,
    fs: Any,
    exec_state: Any,
    on_event: Callable[[BaseEvent], None] | None = None,
) -> str:
    """Execute terminal script and emit output event.

    Args:
        agent_name: Name of the agent (for OutputEvent attribution)
        terminal_script: The terminal script to execute
        fs: FileSystem to execute against
        exec_state: Execution state for event logging
        on_event: Optional callback for event emission

    Returns:
        The stdout from terminal execution

    Raises:
        Exception: Re-raises ParseError or TerminalError after logging output
    """
    from agex.terminal import ParseError, to_script
    from agex.terminal.interpreter.core import execute_script
    from agex.terminal.interpreter.datatypes import TerminalError

    try:
        script = to_script(terminal_script)
        stdout = execute_script(script, fs)

        # Create output event with stdout (no echo of input - ActionEvent has that)
        if stdout:
            output_event = OutputEvent(
                agent_name=agent_name,
                parts=[PrintAction([stdout])],
            )
            add_event_to_log(exec_state, output_event, on_event=on_event)

        return stdout

    except ParseError as e:
        # Syntax error in terminal script
        error_text = f"💥 Terminal parse error: {e}\nCheck your command syntax!"
        error_event = OutputEvent(
            agent_name=agent_name,
            parts=[PrintAction([error_text])],
        )
        add_event_to_log(exec_state, error_event, on_event=on_event)
        raise

    except TerminalError as e:
        # Execution error - include partial output if available
        error_parts = []
        if e.partial_output:
            error_parts.append(e.partial_output)
        error_parts.append(
            f"💥 Terminal error: {e.message}\nAdjust your commands accordingly!"
        )
        error_text = "\n".join(error_parts)

        error_event = OutputEvent(
            agent_name=agent_name,
            parts=[PrintAction([error_text])],
        )
        add_event_to_log(exec_state, error_event, on_event=on_event)
        raise


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
