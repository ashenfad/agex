"""
Processes sandtrap ExecResult back into agex's kvgit state and event system.

Handles:
- Syncing namespace changes back to the kvgit Store
- Detecting variable deletions
- Re-raising _AgentExit signals captured by sandtrap
- Converting modules to pickleable ModuleRef for cross-turn persistence
"""

from __future__ import annotations

import base64
import inspect
import io
import types
from collections.abc import MutableMapping
from typing import Any, Callable

from sandtrap import ExecResult
from sandtrap.wrappers import ModuleRef

from agex.agent.datatypes import TaskContinue, TaskSuccess
from agex.agent.events import OutputEvent
from agex.eval.bridge.namespace import _is_internal_state_key
from agex.eval.objects import ImageAction
from agex.state.log import add_event_to_log


def handle_result(
    result: ExecResult,
    state: MutableMapping[str, Any],
    agent_name: str,
    pre_keys: set[str],
    on_event: Callable[[Any], None] | None = None,
    injected_keys: set[str] | None = None,
    pre_ids: dict[str, int] | None = None,
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
        pre_ids: Object identity snapshot from before execution.  When
                 provided, only variables whose ``id()`` changed (i.e.
                 were reassigned or newly created) are written back to
                 state.  Variables that were merely *referenced* but not
                 reassigned are left to ``safe_commit``'s
                 ``referenced_keys`` path, which re-stages them for
                 byte-level change detection (catching in-place mutations
                 without creating duplicate blobs for unchanged values).

    Raises:
        _AgentExit subclasses: TaskSuccess, TaskFail, TaskContinue, etc.
        Exception: Any regular exception from agent code.
    """
    skip = injected_keys or set()
    ids = pre_ids or {}

    # 1. Sync namespace values back to state — only write variables that
    #    were reassigned (different object identity) or newly created.
    #    Variables with the same id() are untouched by agent code and
    #    don't need to be re-staged; safe_commit's referenced_keys path
    #    will catch any in-place mutations via byte comparison.
    for key, value in result.namespace.items():
        if key.startswith("__") or key in skip:
            continue
        if key in ids and id(value) == ids[key]:
            continue  # Same object — not reassigned
        if isinstance(value, types.ModuleType):
            # Modules can't survive pickle — store a ref that _auto_activate
            # will resolve via __sb_import__ on the next turn.
            state[key] = ModuleRef(value.__name__, getattr(value, "__file__", None))
        else:
            state[key] = value

    # 2. Detect deletions (key was in state before exec, not in namespace after).
    #    Internal bookkeeping keys (__..., _event_...) are never hydrated
    #    into the sandbox, so a well-behaved caller won't list them in
    #    ``pre_keys``. Belt-and-suspenders: still filter them here, so
    #    a buggy caller can't accidentally have us delete the event log's
    #    backing store.
    post_keys = {k for k in result.namespace if not _is_internal_state_key(k)}
    for key in pre_keys - post_keys:
        if _is_internal_state_key(key):
            continue
        if key in state:
            del state[key]

    # 3. Convert print snapshots into OutputEvents
    #    Intercept __AGEX_IMAGE__: prefixed prints and convert to ImageAction
    _IMG_PREFIX = "__AGEX_IMAGE__:"
    for args in result.prints:
        parts = list(args)
        if (
            len(parts) == 1
            and isinstance(parts[0], str)
            and parts[0].startswith(_IMG_PREFIX)
        ):
            try:
                # Defer PIL until we actually need to decode an image —
                # keeps ``import agex`` fast when the agent never prints
                # __AGEX_IMAGE__ markers.
                from PIL import Image  # noqa: PLC0415

                b64 = parts[0][len(_IMG_PREFIX) :]
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                event = OutputEvent(
                    agent_name=agent_name, parts=[ImageAction(image=img)]
                )
            except Exception:
                event = OutputEvent(agent_name=agent_name, parts=parts)
        else:
            event = OutputEvent(agent_name=agent_name, parts=parts)
        add_event_to_log(state, event, on_event=on_event)

    # 4. Convert __outputs__ entries (e.g. view_image) into OutputEvents
    for item in result.namespace.get("__outputs__", []):
        event = OutputEvent(agent_name=agent_name, parts=[item])
        add_event_to_log(state, event, on_event=on_event)

    # 5. Validate TaskSuccess result type (moved from sandbox-side closure
    #    so task_success can be a plain picklable function for cross-process)
    if isinstance(result.error, TaskSuccess):
        _validate_task_result(result.error.result, state)

    # 6. Convert task_continue observations to OutputEvent
    if isinstance(result.error, TaskContinue) and result.error.observations:
        event = OutputEvent(
            agent_name=agent_name, parts=list(result.error.observations)
        )
        add_event_to_log(state, event, on_event=on_event)

    # 7. Re-raise any error captured by sandtrap
    # sandtrap catches ALL BaseException (except KeyboardInterrupt) and puts it
    # in result.error. This includes _AgentExit subclasses (TaskSuccess, TaskFail,
    # TaskContinue, TaskClarify) which are BaseException.
    if result.error is not None:
        raise result.error


def _validate_task_result(result: Any, state: MutableMapping[str, Any]) -> None:
    """Validate task_success result against the expected return type."""
    return_type = state.get("__expected_return_type__")
    if not return_type or return_type is inspect.Parameter.empty:
        return

    from agex.eval.validation import validate_with_sampling

    try:
        validate_with_sampling(result, return_type)
    except Exception as e:
        if (
            hasattr(return_type, "__module__")
            and hasattr(return_type, "__name__")
            and not hasattr(return_type, "__origin__")
        ):
            type_name = return_type.__name__
        else:
            type_name = str(return_type)
        raise TypeError(
            f"Output validation failed. The returned value did not match "
            f"the expected type '{type_name}'.\nDetails: {e}",
        ) from e
