"""
Builds the execution namespace dict for sblite's Sandbox.exec().

Hydrates kvit state values into a plain dict, then injects
task control functions and stateful builtins as closures.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable

from kvit import Store

from agex.agent.datatypes import TaskClarify, TaskContinue, TaskFail, TaskSuccess
from agex.agent.events import OutputEvent
from agex.state.log import add_event_to_log

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent


def build_namespace(
    state: Store,
    agent: "BaseAgent",
    agent_name: str,
    on_event: Callable[[Any], None] | None = None,
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Build the execution namespace from state and builtins.

    Args:
        state: The kvit state to hydrate from.
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
    for key in state.keys():
        if not key.startswith("__"):
            try:
                namespace[key] = state.get(key)
            except Exception:
                continue  # Skip unpicklable or corrupt values
            pre_keys.add(key)

    # 2. Inject task control functions (wrappers that raise, since
    #    in real exec() calling a BaseException subclass constructor
    #    just creates an instance — it doesn't raise automatically)
    def task_success(result=None):
        _validate_task_result(result, state)
        raise TaskSuccess(result)

    def task_fail(message=""):
        raise TaskFail(message)

    def task_clarify(message=""):
        raise TaskClarify(message)

    namespace["task_success"] = task_success
    namespace["task_fail"] = task_fail
    namespace["task_clarify"] = task_clarify
    namespace["task_continue"] = _make_task_continue(state, agent_name, on_event)

    # 3. Inject stateful builtins (print is handled via Sandbox print_handler)
    namespace["view_image"] = _make_view_image(state, agent_name, on_event)
    namespace["help"] = _make_help(agent, agent_name, state, on_event)
    namespace["dir"] = _make_dir(agent, agent_name, state, on_event)

    injected_keys = {
        "task_success",
        "task_fail",
        "task_clarify",
        "task_continue",
        "view_image",
        "help",
        "dir",
    }

    return namespace, pre_keys, injected_keys


def _validate_task_result(result: Any, state: Store) -> None:
    """Validate task_success result against the expected return type."""
    import inspect

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


def make_print_handler(
    state: Store, agent_name: str, on_event: Callable | None
) -> Callable:
    """Create a print handler for sblite's Sandbox.

    This is passed as `print_handler` to `Sandbox.__init__`. sblite wraps it
    with checkpoint enforcement, so we only handle output capture here.

    Captures real Python objects into OutputEvent.parts for budget-aware
    rendering, rather than stringifying into stdout.
    """

    def handler(*args: Any, **kwargs: Any) -> None:
        # Ignore sep/end/file/flush kwargs — we capture objects, not formatted strings
        _do_print(args, state, agent_name, on_event)

    return handler


def _make_task_continue(
    state: Store, agent_name: str, on_event: Callable | None
) -> Callable:
    """Create a task_continue function that optionally prints then raises."""

    def task_continue(*observations: Any) -> None:
        if observations:
            _do_print(observations, state, agent_name, on_event)
        raise TaskContinue()

    return task_continue


def _do_print(
    args: tuple[Any, ...],
    state: Store,
    agent_name: str,
    on_event: Callable | None,
) -> None:
    """Shared print logic: snapshot args and create OutputEvent."""
    try:
        snapped = copy.deepcopy(args)
    except Exception:
        from agex.eval.builtins import _smart_render_for_snapshot

        snapped = tuple(_smart_render_for_snapshot(a) for a in args)

    event = OutputEvent(agent_name=agent_name, parts=list(snapped))
    add_event_to_log(state, event, on_event=on_event)


def _make_view_image(
    state: Store, agent_name: str, on_event: Callable | None
) -> Callable:
    """Create a view_image function closure."""

    def view_image(image: Any, detail: str = "high") -> None:
        from agex.eval.builtins import _view_image_stateful

        _view_image_stateful(
            image, detail, state=state, agent_name=agent_name, on_event=on_event
        )

    return view_image


def _make_help(
    agent: "BaseAgent",
    agent_name: str,
    state: Store,
    on_event: Callable | None,
) -> Callable:
    """Create a help function that inspects agent policy."""

    def help_fn(*args: Any) -> None:
        from agex.eval.builtins import (
            _get_general_help_text,
            _get_help_text,
            _is_allowed_for_help,
        )
        from agex.eval.user_errors import AgexError, AgexTypeError

        if len(args) > 1:
            raise AgexError(f"help() takes at most 1 argument ({len(args)} given)")

        item = args[0] if args else None

        if item is not None and not _is_allowed_for_help(item):
            raise AgexTypeError("help() is only supported for registered resources.")

        doc = _get_help_text(agent, item) if item else _get_general_help_text(agent)
        event = OutputEvent(agent_name=agent_name, parts=[doc])
        add_event_to_log(state, event, on_event=on_event)

    return help_fn


def _make_dir(
    agent: "BaseAgent",
    agent_name: str,
    state: Store,
    on_event: Callable | None,
) -> Callable:
    """Create a dir function that lists available attributes."""

    def dir_fn(*args: Any) -> list[str]:
        from agex.eval.builtins import _is_bound_instance_object
        from agex.eval.objects import (
            AgexModule,
            BoundInstanceObject,
        )
        from agex.eval.user_errors import AgexError
        from agex.eval.utils import get_allowed_attributes_for_instance

        if len(args) > 1:
            raise AgexError(f"dir() takes at most 1 argument ({len(args)} given)")

        obj = args[0] if args else None

        if obj is None:
            attrs = sorted(state.keys())
        elif isinstance(obj, AgexModule):
            ns = agent._policy.namespaces.get(obj.name)
            if ns is None:
                attrs = []
            else:
                from agex.agent.policy.describe import describe_namespace

                desc = describe_namespace(ns, include_low=False)
                attrs = sorted(k for k in desc.keys())
        elif _is_bound_instance_object(obj) and isinstance(obj, BoundInstanceObject):
            methods = list(obj.reg_object.methods.keys())
            properties = list(obj.reg_object.properties.keys())
            attrs = sorted(methods + properties)
        else:
            allowed = get_allowed_attributes_for_instance(agent, obj)
            attrs = sorted(list(allowed))

        final_attrs = [attr for attr in attrs if not attr.startswith("_")]

        event = OutputEvent(agent_name=agent_name, parts=[final_attrs])
        add_event_to_log(state, event, on_event=on_event)

        return final_attrs

    return dir_fn
