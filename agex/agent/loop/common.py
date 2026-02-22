"""
Common helpers, constants, and event factories for the task loop.

This module contains shared logic used by both sync and async task loop implementations.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

from gitkv import Live, Namespaced, Staged
from pydantic import ValidationError
from termish import ParseError, TerminalError, execute_script, to_script

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
from agex.llm.core import LLMResponse, ResponseBuilder, ResponseParseError, StreamToken
from agex.state import (
    ConcurrencyError,
    MergeConflict,
    events,
    is_live_root,
    raw_get,
    raw_remove,
    safe_commit,
)
from agex.state.log import add_event_to_log, get_events_from_log

# Re-export commonly used items for convenience
__all__ = [
    # Constants
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
    "initialize_exec_state",
    "check_for_task_call",
    "strip_namespace_prefix",
    "yield_new_events",
    "maybe_file_event",
    "maybe_add_file_event",
    "apply_optimistic_file_actions",
    "safe_commit",
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
    "MergeConflict",
    "Live",
    "Namespaced",
    "Staged",
    "events",
    "is_live_root",
    "add_event_to_log",
    "get_events_from_log",
    "check_cancellation",
]


def check_cancellation(
    task_name: str,
    versioned_state: Staged | None,
    exec_state: MutableMapping[str, Any],
) -> bool:
    """
    Check if a cancellation sentinel is present for the given task.

    Reads directly from the underlying KV store for Staged state to ensure
    immediate visibility of cancellation requests from other threads/processes.

    Args:
        task_name: Name of the task to check cancellation for
        versioned_state: The Staged state if present, or None
        exec_state: The execution state (Live or Namespaced)

    Returns:
        True if cancellation was detected (and sentinel was cleaned up), False otherwise
    """
    cancel_key = f"__agex_cancel__{task_name}"

    if isinstance(versioned_state, Staged):
        # Read directly from KV store for immediate visibility
        if raw_get(versioned_state, cancel_key):
            # Clean up the sentinel
            raw_remove(versioned_state, cancel_key)
            return True
    else:
        # Live/Namespaced state - check exec_state directly
        if exec_state.get(cancel_key):
            exec_state.pop(cancel_key, None)
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
    state: Staged | Live | Namespaced | None,
    inputs_instance: Any,
    return_type: type,
    session: str = "default",
) -> tuple[Staged | Live | Namespaced, Staged | None]:
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
    versioned_state: Staged | None = None
    exec_state: Staged | Live | Namespaced

    if isinstance(state, Namespaced):
        # Namespaced = someone else owns versioning, we just work within namespace
        exec_state = state
        versioned_state = None
    elif isinstance(state, Staged):
        # Staged = we're responsible for versioning this state
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
        exec_state["inputs"] = inputs_instance
    exec_state["__expected_return_type__"] = return_type

    # Initialize the event log if it doesn't exist
    if "__event_log__" not in exec_state:
        exec_state["__event_log__"] = []

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


def _find_indent_flexible_match(
    search: str, content: str
) -> list[tuple[int, int, str]]:
    """Find matches for search in content with flexible indentation.

    This handles cases where the search and content have the same code structure
    but different absolute indentation levels (e.g., agent sends 2-space indent
    but file uses 4-space or tabs).

    The algorithm:
    1. Find the first non-empty line in search and strip it
    2. Search for that stripped content in the file
    3. For each potential match, verify all lines match when stripped
    4. Return matches with their positions and the actual matched text

    Args:
        search: The search string from the EDIT action
        content: The file content to search in

    Returns:
        List of (start_pos, end_pos, matched_text) tuples for each match found.
        Positions are byte offsets into content.
    """
    search_lines = search.split("\n")
    content_lines = content.split("\n")

    # Find first non-empty line in search for anchoring
    anchor_stripped = None
    anchor_idx = 0
    for idx, line in enumerate(search_lines):
        stripped = line.strip()
        if stripped:
            anchor_stripped = stripped
            anchor_idx = idx
            break

    if anchor_stripped is None:
        return []

    # Build list of stripped search lines for comparison
    search_stripped = [line.strip() for line in search_lines]

    matches = []

    # Search for anchor line in content
    for i, content_line in enumerate(content_lines):
        content_line_stripped = content_line.strip()

        # Check if this line matches our anchor (accounting for trailing ws)
        if content_line_stripped != anchor_stripped:
            continue

        # Potential match - calculate where the full block would start
        start_line = i - anchor_idx
        if start_line < 0:
            continue

        # Check if we have enough lines
        end_line = start_line + len(search_lines)
        if end_line > len(content_lines):
            continue

        # Verify all lines match when stripped
        match = True
        for j, search_line_stripped in enumerate(search_stripped):
            content_idx = start_line + j
            content_stripped = content_lines[content_idx].strip()

            # Both should be empty or both should have same stripped content
            # Also handle trailing whitespace flexibility
            if search_line_stripped.rstrip() != content_stripped.rstrip():
                match = False
                break

        if match:
            # Calculate byte positions
            # Sum lengths of all lines before start_line, plus newlines
            start_pos = sum(len(content_lines[k]) + 1 for k in range(start_line))

            # Calculate end position (end of the last matched line)
            matched_lines = content_lines[start_line:end_line]
            matched_text = "\n".join(matched_lines)

            # End position is start + length of matched text
            end_pos = start_pos + len(matched_text)

            matches.append((start_pos, end_pos, matched_text))

    return matches


def _adjust_replacement_indent(replacement: str, search: str, matched_text: str) -> str:
    """Adjust replacement indentation to match the target file's style.

    When we match with flexible indentation, the replacement content needs
    its indentation adjusted to fit naturally into the target file.

    Args:
        replacement: The replacement text from the EDIT action
        search: The original search text (to determine agent's indent baseline)
        matched_text: The actual text that was matched in the file

    Returns:
        Replacement text with indentation adjusted to match the file's style
    """
    search_lines = search.split("\n")
    matched_lines = matched_text.split("\n")
    replacement_lines = replacement.split("\n")

    def get_base_indent_info(lines: list[str]) -> tuple[int, str, int]:
        """Get base indentation info from first non-empty line.

        Returns:
            (indent_in_spaces, indent_char, raw_char_count)
            - indent_in_spaces: equivalent space count (tabs count as 4)
            - indent_char: '\t' if tabs used, ' ' otherwise
            - raw_char_count: actual character count of leading whitespace
        """
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                leading = line[: len(line) - len(stripped)]
                indent_char = "\t" if "\t" in leading else " "
                # Calculate equivalent spaces (tabs = 4 spaces each)
                indent_in_spaces = leading.count("\t") * 4 + leading.count(" ")
                return indent_in_spaces, indent_char, len(leading)
        return 0, " ", 0

    search_base_indent, _, _ = get_base_indent_info(search_lines)
    target_base_indent, target_indent_char, _ = get_base_indent_info(matched_lines)
    replacement_base_indent, _, _ = get_base_indent_info(replacement_lines)

    # Calculate the indent adjustment needed (in equivalent spaces)
    # Heuristic: if replacement base indent matches search base indent,
    # assume agent wrote replacement relative to search context
    if replacement_base_indent == search_base_indent:
        indent_delta = target_base_indent - search_base_indent
    else:
        # Agent used different indent in replacement - shift to match target
        indent_delta = target_base_indent - replacement_base_indent

    # Adjust each line in replacement
    adjusted = []
    for line in replacement_lines:
        stripped = line.lstrip()
        if not stripped:
            # Preserve empty lines (but strip any whitespace for cleanliness)
            adjusted.append("")
        else:
            # Calculate current indent in equivalent spaces
            current_leading = line[: len(line) - len(stripped)]
            current_indent = current_leading.count("\t") * 4 + current_leading.count(
                " "
            )
            new_indent = max(0, current_indent + indent_delta)

            # Use target indent style
            if target_indent_char == "\t":
                # Convert to tabs (4 spaces per tab)
                tabs = new_indent // 4
                spaces = new_indent % 4
                new_leading = "\t" * tabs + " " * spaces
            else:
                new_leading = " " * new_indent

            # Preserve trailing whitespace from original replacement
            trailing = (
                line[len(line) - len(line.rstrip()) :] if line.rstrip() != line else ""
            )
            adjusted.append(new_leading + stripped + trailing)

    return "\n".join(adjusted)


def apply_optimistic_file_actions(
    agent: Any,
    llm_response: LLMResponse,
    fs: Any,
    exec_state: MutableMapping[str, Any],
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

            target_fs.write(path, content.encode("utf-8"), mode=fs_mode)

        elif isinstance(action, EditAction):
            path = action.path

            # Read existing file
            try:
                existing_content = target_fs.read(path).decode("utf-8")
            except FileNotFoundError:
                raise ResponseParseError(f"File not found for EDIT: {path}")

            # Matching strategy (try in order, stop at first success):
            # 1. Exact match
            # 2. Trailing whitespace flexible match
            # 3. Indent-flexible match (different absolute indentation)

            match_mode = "exact"
            count = existing_content.count(action.search)
            indent_matches: list[tuple[int, int, str]] = []

            if count == 0:
                # Exact match failed - try trailing whitespace flexible matching
                pattern = _build_trailing_ws_pattern(action.search)
                regex_matches = list(pattern.finditer(existing_content))
                count = len(regex_matches)
                match_mode = "trailing_ws"

                if count == 0:
                    # Trailing ws match failed - try indent-flexible matching
                    indent_matches = _find_indent_flexible_match(
                        action.search, existing_content
                    )
                    count = len(indent_matches)
                    match_mode = "indent_flexible"

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
            if match_mode == "indent_flexible":
                # Indent-flexible matching - need to adjust replacement indentation
                # Process matches in reverse order to preserve positions
                matches_to_apply = (
                    indent_matches if action.match_all else indent_matches[:1]
                )
                new_content = existing_content

                for start_pos, end_pos, matched_text in reversed(matches_to_apply):
                    # Adjust replacement content's indentation to match the file
                    adjusted_content = _adjust_replacement_indent(
                        action.content, action.search, matched_text
                    )

                    if action.operation == "insert-after":
                        replacement = matched_text + adjusted_content
                    elif action.operation == "insert-before":
                        replacement = adjusted_content + matched_text
                    else:  # "replace"
                        replacement = adjusted_content

                    new_content = (
                        new_content[:start_pos] + replacement + new_content[end_pos:]
                    )

            elif match_mode == "trailing_ws":
                # Trailing whitespace flexible matching - use regex replacement
                pattern = _build_trailing_ws_pattern(action.search)

                def make_replacement(match: re.Match) -> str:
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
    exec_state: MutableMapping[str, Any],
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

    fs_metadata_after = fs.get_metadata_snapshot()
    file_event = maybe_file_event(agent_name, fs_metadata_before, fs_metadata_after)

    if file_event:
        # Add to log WITHOUT on_event - we'll emit after commit
        add_event_to_log(exec_state, file_event)

    return file_event
