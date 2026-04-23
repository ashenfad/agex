"""
Builds the execution namespace dict for sandtrap's Sandbox.exec().

Hydrates kvgit state values into a plain dict, then injects
task control functions and view_image.
"""

from __future__ import annotations

import builtins as _builtins
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Callable

from agex.agent.datatypes import TaskClarify, TaskFail, TaskSuccess
from agex.eval.bridge.policy import _current_emission_id
from agex.eval.objects import ImageAction

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent


# Prefixes for state keys that are internal bookkeeping (event log,
# framework metadata) and must never be hydrated into the agent's
# sandbox namespace. Exposing them is both a privacy leak (the agent
# could see its own prior events) and a correctness hazard: if a key
# ends up in ``pre_keys`` but not in the sandbox's post-exec namespace,
# ``handle_result`` treats it as "user deleted it" and removes it from
# state. That silently corrupted ``__event_log__`` in the wild —
# events were referenced by the log but their backing keys had been
# deleted on a subsequent agent turn.
_INTERNAL_STATE_PREFIXES = ("__", "_event_")


def _is_internal_state_key(key: str) -> bool:
    return any(key.startswith(p) for p in _INTERNAL_STATE_PREFIXES)


def build_namespace(
    state: MutableMapping[str, Any],
    agent: "BaseAgent",
    agent_name: str,
    on_event: Callable[[Any], None] | None = None,
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Build the execution namespace from state and builtins.

    Args:
        state: The kvgit state to hydrate from.
        agent: The agent providing policy context.
        agent_name: Name of the agent (for event attribution).
        on_event: Optional event callback.

    Returns:
        A tuple of (namespace_dict, pre_keys, injected_keys) where pre_keys
        is the set of user-visible state keys before execution (for deletion
        tracking) and injected_keys is the set of bridge-injected names that
        should not be synced back to state.
    """
    namespace: dict[str, Any] = {}
    pre_keys: set[str] = set()

    # 1. Hydrate from state (skip internal keys)
    from agex.agent.datatypes import UnpicklableMarker, UnpicklableVariableError

    for key in state.keys():
        if _is_internal_state_key(key):
            continue
        try:
            namespace[key] = state.get(key)
        except UnpicklableVariableError as exc:
            # Place marker in namespace so the agent gets a descriptive
            # error on access rather than a bare NameError.
            if exc.marker is not None:
                exc.marker.variable_name = key
                namespace[key] = exc.marker
            else:
                namespace[key] = UnpicklableMarker(
                    variable_name=key,
                    type_name="<unknown>",
                    original_exception=str(exc),
                )
        except Exception:
            continue  # Skip corrupt values
        pre_keys.add(key)

    # 2. Inject task control functions — module-level so they're picklable
    #    for cross-process isolation. Validation happens in handle_result.
    namespace["task_success"] = _task_success
    namespace["task_fail"] = _task_fail
    namespace["task_clarify"] = _task_clarify

    # 3. Inject __outputs__ list and picklable view_image.
    #    view_image appends to __outputs__; handle_result drains it into events.
    outputs: list = []
    namespace["__outputs__"] = outputs
    namespace["view_image"] = _ViewImage(outputs)

    # 4. Override dir() to hide internal keys (_event_*, __dunder__, sandtrap
    #    injections) while exposing user state and task control functions.
    _visible_names = {
        "task_success",
        "task_fail",
        "task_clarify",
        "view_image",
    }
    # Add user state keys (excluding _event_* and other internal prefixed keys)
    _visible_names |= {k for k in pre_keys if not k.startswith("_")}
    namespace["dir"] = _AgentDir(_visible_names)

    injected_keys = {
        "task_success",
        "task_fail",
        "task_clarify",
        "view_image",
        "__outputs__",
        "dir",
    }

    return namespace, pre_keys, injected_keys


def _task_success(result=None):
    """Signal task completion. Validation happens in handle_result."""
    raise TaskSuccess(result)


def _task_fail(message=""):
    """Signal task failure."""
    raise TaskFail(message)


def _task_clarify(message=""):
    """Signal that more information is needed."""
    raise TaskClarify(message)


class _AgentDir:
    """Custom dir() that shows user-relevant names only.

    Hides internal keys (_event_*, __dunder__, sandtrap-injected classes)
    while exposing user state variables, task control functions, and any
    variables the agent defines during execution.
    """

    _real_dir = _builtins.dir

    def __init__(self, visible_names: set[str]):
        self._visible_names = visible_names

    def __call__(self, obj=_builtins, /):
        if obj is not _builtins:  # called with an argument
            return self._real_dir(obj)
        import sys

        caller_locals = sys._getframe(1).f_locals
        # Start with locals defined by the agent during execution
        names = {k for k in caller_locals if not k.startswith("_")}
        # Add pre-approved names (user state + task control builtins)
        names |= self._visible_names
        return sorted(names)


class _ViewImage:
    """Picklable view_image() — appends to __outputs__ for cross-process support.

    Reads the current ``emission_id`` contextvar so the :class:`ImageAction`
    traces back to the PythonEmission whose execution called it.
    """

    def __init__(self, outputs: list):
        self._outputs = outputs

    def __call__(self, image: Any, detail: str = "high") -> None:
        if detail not in ("low", "high"):
            raise ValueError("detail must be 'low' or 'high'")
        emission_id = _current_emission_id.get()
        self._outputs.append(
            ImageAction(image=image, detail=detail, emission_id=emission_id)
        )
