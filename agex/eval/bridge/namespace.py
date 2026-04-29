"""
Builds the execution namespace dict for sandtrap's Sandbox.exec().

Each ``python_action`` runs as a fresh script, so ``build_namespace`` is
called once per emission and produces a clean dict containing only the
bridge-injected names (task control functions, view_image, ``dir``).
Sandtrap layers registered modules and functions on top via the policy.
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


# Internal-key prefixes still consulted by ``handle_result`` when it
# filters the post-exec namespace; lives here for the moment so the
# import stays stable while the bridge is rewired.
_INTERNAL_STATE_PREFIXES = ("__", "_event_")


def _is_internal_state_key(key: str) -> bool:
    return any(key.startswith(p) for p in _INTERNAL_STATE_PREFIXES)


def build_namespace(
    state: MutableMapping[str, Any],
    agent: "BaseAgent",
    agent_name: str,
    on_event: Callable[[Any], None] | None = None,
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Build a fresh execution namespace for one ``python_action``.

    Each call returns an independent dict — there is no cross-emission
    state continuity. ``state`` is unused under the stateless contract
    and remains in the signature only until the call site is updated.

    Args:
        state: Unused. Retained pending call-site update.
        agent: The agent providing policy context.
        agent_name: Name of the agent (for event attribution).
        on_event: Optional event callback.

    Returns:
        A tuple of (namespace_dict, pre_keys, injected_keys). ``pre_keys``
        is always empty under the new contract and will be removed when
        ``handle_result`` no longer needs it.
    """
    del state  # unused under stateless contract; will be dropped from signature

    namespace: dict[str, Any] = {}

    # Inject task control functions — module-level so they're picklable
    # for cross-process isolation. Validation happens in handle_result.
    namespace["task_success"] = _task_success
    namespace["task_fail"] = _task_fail
    namespace["task_clarify"] = _task_clarify

    # Inject __outputs__ list and picklable view_image.
    # view_image appends to __outputs__; handle_result drains it into events.
    outputs: list = []
    namespace["__outputs__"] = outputs
    namespace["view_image"] = _ViewImage(outputs)

    injected_keys = {
        "task_success",
        "task_fail",
        "task_clarify",
        "view_image",
        "__outputs__",
        "dir",
    }

    # Override dir() to hide sandtrap-injected internals while exposing
    # task control functions and any names the agent defines during
    # execution. Registered functions and modules show up via sandtrap's
    # own scoping; this list is just the names we put in the namespace
    # dict directly.
    namespace["dir"] = _AgentDir(injected_keys - {"__outputs__", "dir"})

    return namespace, set(), injected_keys


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

    Hides sandtrap-injected internals (``__st_*`` gates, dunder names)
    while exposing the task control terminators plus whatever names the
    agent defined during this action.
    """

    _real_dir = _builtins.dir

    def __init__(self, visible_names: set[str]):
        self._visible_names = visible_names

    def __call__(self, obj=_builtins, /):
        if obj is not _builtins:  # called with an argument
            return self._real_dir(obj)
        import sys

        caller_locals = sys._getframe(1).f_locals
        # Start with locals defined by the agent during this action
        names = {k for k in caller_locals if not k.startswith("_")}
        # Add the bridge-injected terminators
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
