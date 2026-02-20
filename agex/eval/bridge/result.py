"""
Processes sblite ExecResult back into agex's kvit state and event system.

Handles:
- Syncing namespace changes back to the kvit Store
- Detecting variable deletions
- Re-raising _AgentExit signals captured by sblite
"""

from __future__ import annotations

from typing import Any, Callable

from kvit import Store
from sblite import ExecResult


def handle_result(
    result: ExecResult,
    state: Store,
    agent_name: str,
    pre_keys: set[str],
    on_event: Callable[[Any], None] | None = None,
) -> None:
    """Process an ExecResult: sync state and re-raise errors.

    Args:
        result: The ExecResult from sblite Sandbox.exec().
        state: The kvit state to sync changes back to.
        agent_name: Agent name for event attribution.
        pre_keys: Set of user-visible state keys before execution
                  (for deletion detection).
        on_event: Optional event callback.

    Raises:
        _AgentExit subclasses: TaskSuccess, TaskFail, TaskContinue, etc.
        Exception: Any regular exception from agent code.
    """
    # 1. Sync namespace values back to state
    for key, value in result.namespace.items():
        if not key.startswith("__"):
            state.set(key, value)

    # 2. Detect deletions (key was in state before exec, not in namespace after)
    post_keys = {k for k in result.namespace if not k.startswith("__")}
    for key in pre_keys - post_keys:
        if key in state:
            state.remove(key)

    # 3. Re-raise any error captured by sblite
    # sblite catches ALL BaseException (except KeyboardInterrupt) and puts it
    # in result.error. This includes _AgentExit subclasses (TaskSuccess, TaskFail,
    # TaskContinue, TaskClarify) which are BaseException.
    if result.error is not None:
        raise result.error
