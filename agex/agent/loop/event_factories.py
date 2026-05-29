"""Event factories and terminal execution for the task loop."""

from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from monkeyfs import MountFS
from termish import ParseError, TerminalError, execute_script, to_script

from agex.agent.events import (
    ActionEvent,
    BaseEvent,
    ClarifyEvent,
    FailEvent,
    OutputEvent,
    PermissionEvent,
    PermissionRequestEvent,
    SuccessEvent,
    SystemNoteEvent,
    TaskStartEvent,
)
from agex.eval.objects import PrintAction
from agex.llm.core import LLMResponse
from agex.state.log import add_event_to_log

if TYPE_CHECKING:
    from agex.terminal import TerminalCommandRegistration

# Task control guidance message (shown when agent forgets to signal completion).
# Python completing without a terminator implicitly continues the turn — this
# reminder nudges the agent to wrap up with an explicit terminator when
# iterations are running long.
TASK_CONTROL_GUIDANCE = (
    "💡 **Silent turn** — your python_action ran without printing "
    "anything, so there's nothing to observe next turn.  If you're "
    "done, finish explicitly inside python_action:\n\n"
    "• `task_success(result)` — complete the task with your final answer\n"
    "• `task_fail(message)` — if you cannot complete the task\n"
    "• `task_clarify(message)` — if you need more information from the caller\n\n"
    "Otherwise keep going — your turn continues normally on the next iteration."
)

# Shown when a turn completed with only text/thinking — no tool_use.
# Prose alone doesn't advance the task; the model needs to pick up a
# lever (python_action / write_file / edit_file / terminal_action) to
# make progress or finish.
NO_PROGRESS_GUIDANCE = (
    "⚠️ **No tools called last turn** — plain text doesn't execute "
    "anything or finish the task.  If you know what to do, call a "
    "tool this turn (write_file / edit_file / terminal_action / "
    "python_action).  If you're done, call `task_success(result)` "
    "inside python_action.  If you're stuck or need more info, call "
    "`task_clarify(message)` or `task_fail(message)`."
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
        emissions=list(llm_response.emissions),
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


def create_permission_request_event(
    agent_name: str, scopes: set[str], task_name: str, reason: str | None = None
) -> PermissionRequestEvent:
    """Create a PermissionRequestEvent (a task suspended to request scope(s))."""
    return PermissionRequestEvent(
        agent_name=agent_name, scopes=scopes, task_name=task_name, reason=reason
    )


def create_permission_event(
    agent_name: str,
    granted: list[str] | None = None,
    denied: list[str] | None = None,
    revoked: list[str] | None = None,
    note: str | None = None,
) -> PermissionEvent:
    """Create a PermissionEvent (host granted/denied/revoked scopes)."""
    return PermissionEvent(
        agent_name=agent_name,
        granted=granted or [],
        denied=denied or [],
        revoked=revoked or [],
        note=note,
    )


def create_error_output(
    agent_name: str,
    exception: Exception,
    emission_id: str | None = None,
) -> OutputEvent:
    """Create an OutputEvent for an evaluation error."""
    return OutputEvent(
        agent_name=agent_name,
        parts=[
            PrintAction(
                args=(f"💥 {type(exception).__name__}: {exception}",),
                emission_id=emission_id,
            )
        ],
    )


def build_terminal_commands(
    agent: Any, fs: Any, state: Any = None, vfs: Any = None
) -> dict:
    """Build the injected commands dict for termish execution.

    ``python`` is always available (core capability and reserved name).
    User-registered commands (via ``agent.terminal(...)`` or, for
    internal use, ``agent._terminal_command_factory(...)``) are added
    on top.  Termish builtins (ls, cat, grep, ...) are NOT in the
    dict — termish loads them itself; user registrations with those
    names override them per termish's existing contract.
    """
    from agex.python_cli import make_python_handler

    commands: dict = {"python": make_python_handler(agent, fs)}

    # User-registered commands (agent.terminal + the internal
    # _terminal_command_factory).  Includes ``git`` when
    # ``register_git(agent)`` has been called — register_git wires
    # through the internal factory API.
    registrations: dict[str, TerminalCommandRegistration] = getattr(
        agent, "_terminal_commands", {}
    )
    for name, reg in registrations.items():
        if name in commands:
            # Reserved names like "python" — registration already
            # raises on collision, but defend here too.
            continue
        commands[name] = _build_termish_handler(reg, fs, state, vfs)

    return commands


def _build_termish_handler(
    reg: "TerminalCommandRegistration",
    fs: Any,
    state: Any,
    vfs: Any,
) -> Callable[..., Any]:
    """Adapt a TerminalCommandRegistration to a termish CommandFunc.

    Constructs a fresh per-invocation closure that:
    - For ``kind="simple"``: builds a TerminalContext from termish's
      CommandContext and dispatches the user's handler.
    - For ``kind="factory"``: invokes the user's factory once with a
      TerminalRuntime, returns the resulting CommandFunc directly.

    Per-action ``state`` and ``vfs`` are captured by the closure (or
    via the factory's TerminalRuntime), so each terminal_action sees
    fresh runtime values without re-registration.
    """
    from agex.terminal import TerminalContext, TerminalRuntime

    if reg.kind == "factory":
        rt = TerminalRuntime(fs=fs, state=state, vfs=vfs)
        # The factory itself returns a termish CommandFunc — pass through.
        return reg.handler(rt)

    # kind == "simple": wrap the user's TerminalContext-shape handler
    # into a termish CommandFunc that translates from CommandContext.
    handler = reg.handler

    def termish_command(cmd_ctx):
        rt_ctx = TerminalContext(
            args=cmd_ctx.args,
            stdin=cmd_ctx.stdin,
            stdout=cmd_ctx.stdout,
            fs=cmd_ctx.fs,
        )
        return handler(rt_ctx)

    return termish_command


def execute_terminal(
    agent_name: str,
    terminal_script: str,
    fs: Any,
    exec_state: MutableMapping[str, Any],
    on_event: Callable[[BaseEvent], None] | None = None,
    commands: dict | None = None,
    emission_id: str | None = None,
) -> str:
    """Execute terminal script and emit output event.

    Args:
        agent_name: Name of the agent (for OutputEvent attribution)
        terminal_script: The terminal script to execute
        fs: FileSystem to execute against
        exec_state: Execution state for event logging
        on_event: Optional callback for event emission
        commands: Optional injected command handlers (e.g. python, git)
        emission_id: Stamped onto PrintAction parts so the renderer
            pairs terminal output back to this emission's tool_use.

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
                parts=[PrintAction(args=(stdout,), emission_id=emission_id)],
            )
            add_event_to_log(exec_state, output_event, on_event=on_event)

        return stdout

    except ParseError as e:
        # Syntax error in terminal script
        error_text = f"💥 Terminal parse error: {e}\nCheck your command syntax!"
        error_event = OutputEvent(
            agent_name=agent_name,
            parts=[PrintAction(args=(error_text,), emission_id=emission_id)],
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
            parts=[PrintAction(args=(error_text,), emission_id=emission_id)],
        )
        add_event_to_log(exec_state, error_event, on_event=on_event)
        raise


def create_guidance_output(
    agent_name: str,
    emission_id: str | None = None,
) -> OutputEvent:
    """Create an OutputEvent with task control guidance."""
    return OutputEvent(
        agent_name=agent_name,
        parts=[PrintAction(args=(TASK_CONTROL_GUIDANCE,), emission_id=emission_id)],
    )


def create_no_progress_guidance(agent_name: str) -> SystemNoteEvent:
    """Create a :class:`SystemNoteEvent` nudging the agent to use a tool.

    Emitted when a turn produced only text / thinking emissions — no
    python_action, terminal_action, write_file, or edit_file.  Text
    alone can't finish the task; the agent needs a lever.
    """
    return SystemNoteEvent(agent_name="System", message=NO_PROGRESS_GUIDANCE)


def create_unsaved_warning(
    agent_name: str,
    unsaved_keys: list[str],
    namespace_prefix: str,
    emission_id: str | None = None,
) -> OutputEvent:
    """Create an OutputEvent warning about unsaved variables."""
    agent_visible_keys = strip_namespace_prefix(unsaved_keys, namespace_prefix)
    warning_message = (
        f"⚠️ Could not save the following variables because they "
        f"are not serializable: {', '.join(agent_visible_keys)}"
    )
    return OutputEvent(
        agent_name=agent_name,
        parts=[PrintAction(args=(warning_message,), emission_id=emission_id)],
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
