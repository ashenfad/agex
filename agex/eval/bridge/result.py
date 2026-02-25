"""
Processes sandtrap ExecResult back into agex's kvgit state and event system.

Handles:
- Syncing namespace changes back to the kvgit Store
- Detecting variable deletions
- Re-raising _AgentExit signals captured by sandtrap
- Converting modules to pickleable ModuleRef for cross-turn persistence
"""

from __future__ import annotations

import types
from collections.abc import MutableMapping
from typing import Any, Callable

from sandtrap import ExecResult
from sandtrap.wrappers import ModuleRef


def handle_result(
    result: ExecResult,
    state: MutableMapping[str, Any],
    agent_name: str,
    pre_keys: set[str],
    on_event: Callable[[Any], None] | None = None,
    injected_keys: set[str] | None = None,
) -> None:
    """Process an ExecResult: sync state and re-raise errors.

    Args:
        result: The ExecResult from sandtrap Sandbox.exec().
        state: The kvgit state to sync changes back to.
        agent_name: Agent name for event attribution.
        pre_keys: Set of user-visible state keys before execution
                  (for deletion detection).
        on_event: Optional event callback.
        injected_keys: Set of bridge-injected names (task_success, etc.)
                       that should not be synced back to state.

    Raises:
        _AgentExit subclasses: TaskSuccess, TaskFail, TaskContinue, etc.
        Exception: Any regular exception from agent code.
    """
    skip = injected_keys or set()

    # 1. Sync namespace values back to state
    for key, value in result.namespace.items():
        if not key.startswith("__") and key not in skip:
            if isinstance(value, types.ModuleType):
                # Modules can't survive pickle — store a ref that _auto_activate
                # will resolve via __sb_import__ on the next turn.
                state[key] = ModuleRef(value.__name__, getattr(value, "__file__", None))
            else:
                state[key] = value

    # 2. Detect deletions (key was in state before exec, not in namespace after)
    post_keys = {k for k in result.namespace if not k.startswith("__")}
    for key in pre_keys - post_keys:
        if key in state:
            del state[key]

    # 3. Re-raise any error captured by sandtrap
    # sandtrap catches ALL BaseException (except KeyboardInterrupt) and puts it
    # in result.error. This includes _AgentExit subclasses (TaskSuccess, TaskFail,
    # TaskContinue, TaskClarify) which are BaseException.
    if result.error is not None:
        raise result.error
