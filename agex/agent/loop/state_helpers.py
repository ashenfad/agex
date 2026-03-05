"""State initialization and control flow helpers for the task loop."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Callable

from kvgit import Namespaced, Staged
from monkeyfs import MountFS

from agex.fs.chapters_vfs import create_chapters_fs
from agex.state import events, get_root, raw_get, raw_remove
from agex.state.live import Live
from agex.state.log import get_events_from_log

if TYPE_CHECKING:
    from agex.llm.core import LLMResponse


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


def mount_chapters_overlay(fs: Any, state: Any) -> None:
    """Mount or update the /chapters overlay on a MountFS.

    Reads events from state and builds a read-only VFS from any ChapterEvents.
    No-op if fs is not a MountFS or if there are no chapters.
    """
    if not isinstance(fs, MountFS):
        return

    all_events = get_events_from_log(state)
    chapters_overlay = create_chapters_fs(all_events)
    if chapters_overlay is not None:
        try:
            fs.unmount("/chapters")
        except ValueError:
            pass
        fs.mount("/chapters", chapters_overlay)


def prepare_task_loop(
    agent: Any,
    state: Staged | Namespaced | None,
    session: str,
) -> tuple[Staged | None, Any, dict]:
    """Resolve versioned_state, filesystem, and metadata snapshot for a task loop.

    Identical preamble used by both _run_task_loop and _arun_task_loop.

    Returns:
        (versioned_state, fs, fs_metadata_before)
    """
    versioned_state: Staged | None = None
    if isinstance(state, Staged):
        versioned_state = state
    elif isinstance(state, Namespaced):
        base = get_root(state)
        if isinstance(base, Staged):
            versioned_state = base

    if agent._fs_config:
        fs, _ = agent._get_fs_backend(session)
        fs_metadata_before = fs.get_metadata_snapshot()

        # Wrap in MountFS if chaptering is enabled (for /chapters overlay)
        if agent.log_high_water_tokens is not None:
            if not isinstance(fs, MountFS):
                fs = MountFS(fs)

            # Mount chapters from previous tasks (if any)
            if state is not None:
                mount_chapters_overlay(fs, state)
    else:
        fs = None
        fs_metadata_before = {}

    return versioned_state, fs, fs_metadata_before


def clear_stale_cancel(
    task_name: str,
    versioned_state: Staged | None,
    exec_state: MutableMapping[str, Any],
) -> None:
    """Clear any stale cancellation signal from a previous run.

    Handles the race condition where a cancel arrives just as the previous
    task finishes — we don't want it to immediately cancel this fresh task.
    """
    cancel_key = f"__agex_cancel__{task_name}"
    if isinstance(versioned_state, Staged):
        raw_remove(versioned_state, cancel_key)
    else:
        exec_state.pop(cancel_key, None)


def process_llm_response(
    llm_response: "LLMResponse",
    strip_fence_fn: Callable[[str], str],
    exec_state: MutableMapping[str, Any],
    accumulated_refs: set[str],
) -> str:
    """Post-process an LLM response: strip fences and find referenced state keys.

    Returns:
        The code string to evaluate (may be empty).
    """
    llm_response.code = strip_fence_fn(llm_response.code)
    code_to_evaluate = llm_response.code
    if code_to_evaluate:
        from sandtrap import find_refs

        try:
            accumulated_refs |= find_refs(code_to_evaluate, namespace=exec_state)
        except Exception:
            pass
    return code_to_evaluate
