"""Event factories and terminal execution for the task loop."""

from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

from monkeyfs import MountFS
from termish import ParseError, TerminalError, execute_script, to_script

from agex.agent.events import (
    ActionEvent,
    BaseEvent,
    ClarifyEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    SystemNoteEvent,
    TaskStartEvent,
)
from agex.eval.objects import PrintAction
from agex.llm.core import LLMResponse
from agex.state.log import add_event_to_log

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


def strip_namespace_prefix(keys: list[str], namespace_prefix: str) -> list[str]:
    """Strip namespace prefix from keys for user-facing messages."""
    result = []
    for key in keys:
        if key.startswith(namespace_prefix):
            result.append(key[len(namespace_prefix) :])
        else:
            result.append(key)
    return result


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
        report=llm_response.report,
        code=llm_response.code,
        terminal=llm_response.terminal,
        file_actions=llm_response.file_actions,
        source=source,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
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
        parts=[PrintAction([f"💥 {type(exception).__name__}: {exception}"])],
    )


def build_terminal_commands(
    agent: Any, fs: Any, state: Any = None, vfs: Any = None
) -> dict:
    """Build the injected commands dict for termish execution.

    ``python`` is always available (core capability).
    ``git`` is available when the agent has the git skill registered.

    Returns an empty dict if no handlers are applicable.
    """
    from agex.python_cli import make_python_handler

    commands: dict = {"python": make_python_handler(agent, fs)}

    # git is opt-in via register_git(agent)
    skill_names = {name for name, _ in getattr(agent, "_skills", [])}
    if "git" in skill_names:
        from agex.git_cli import make_git_handler

        vkv = getattr(state, "_versioned", None) if state is not None else None
        if vkv is not None:
            # Extract the raw VirtualFS from a MountFS wrapper if needed.
            # The git handler needs the VFS for key encoding/decoding.
            raw_vfs = vfs
            if hasattr(vfs, "_base"):
                raw_vfs = vfs._base  # MountFS wraps a base FS
            commands["git"] = make_git_handler(vkv, state=state, vfs=raw_vfs)

    return commands


def execute_terminal(
    agent_name: str,
    terminal_script: str,
    fs: Any,
    exec_state: MutableMapping[str, Any],
    on_event: Callable[[BaseEvent], None] | None = None,
    commands: dict | None = None,
) -> str:
    """Execute terminal script and emit output event.

    Args:
        agent_name: Name of the agent (for OutputEvent attribution)
        terminal_script: The terminal script to execute
        fs: FileSystem to execute against
        exec_state: Execution state for event logging
        on_event: Optional callback for event emission
        commands: Optional injected command handlers (e.g. python, git)

    Returns:
        The stdout from terminal execution

    Raises:
        Exception: Re-raises ParseError or TerminalError after logging output
    """
    try:
        script = to_script(terminal_script)
        stdout = execute_script(script, fs, commands=commands)

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
):
    """Check for file changes and add FileEvent to log if needed.

    This should be called BEFORE commit so the FileEvent is included in the commit.
    The FileEvent should be yielded to the caller, but NOT emitted via on_event yet.
    Emission happens after commit.

    Args:
        fs: The filesystem instance (or None if no fs)
        fs_metadata_before: Snapshot of file metadata before execution
        exec_state: The execution state to add the event to
        agent_name: Name of the agent

    Returns:
        The FileEvent if created, None otherwise
    """

    if not fs:
        return None

    base_fs = fs._base if isinstance(fs, MountFS) else fs
    fs_metadata_after = base_fs.get_metadata_snapshot()
    file_event = maybe_file_event(agent_name, fs_metadata_before, fs_metadata_after)

    if file_event:
        # Add to log WITHOUT on_event - we'll emit after commit
        add_event_to_log(exec_state, file_event)

    return file_event
