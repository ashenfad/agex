"""
Translates agex's AgentPolicy into a sblite Policy.

Maps agex Namespace registrations (module, instance, virtual/__main__)
to sblite's policy.fn(), policy.cls(), and policy.module() calls.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any, Callable

from sblite.policy import MemberSpec as SbliteMemberSpec
from sblite.policy import Policy

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.agent.datatypes import MemberSpec as AgexMemberSpec

# Context variables for propagating session/on_event to sub-agent task calls.
# Set by execute_sandboxed/aexecute_sandboxed before running sandbox code.
_current_session: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_session", default="default"
)
_current_on_event: contextvars.ContextVar[Callable[[Any], None] | None] = (
    contextvars.ContextVar("_current_on_event", default=None)
)


def _translate_configure(
    agex_configure: dict[str, "AgexMemberSpec"],
) -> dict[str, SbliteMemberSpec]:
    """Convert agex MemberSpec configure dict to sblite MemberSpec configure dict.

    Drops agex-only fields (visibility, docstring, constructable) and keeps
    the security-relevant fields (host_fs_access, network_access).
    """
    result: dict[str, SbliteMemberSpec] = {}
    for name, spec in agex_configure.items():
        result[name] = SbliteMemberSpec(
            host_fs_access=getattr(spec, "host_fs_access", False),
            network_access=getattr(spec, "network_access", False),
        )
    return result


def translate_policy(agent: "BaseAgent", timeout: float | None = None) -> Policy:
    """Convert an agent's policy registrations into a sblite Policy.

    Args:
        agent: The agent whose policy to translate.
        timeout: Execution timeout in seconds. Defaults to agent.eval_timeout_seconds.

    Returns:
        A sblite Policy with equivalent registrations.
    """
    effective_timeout = timeout if timeout is not None else agent.eval_timeout_seconds
    policy = Policy(timeout=effective_timeout)

    for ns_name, ns in agent._policy.namespaces.items():
        if ns_name == "__main__" and ns.kind == "virtual":
            _translate_main_namespace(policy, ns, agent)
        elif ns.kind == "module":
            _translate_module_namespace(policy, ns)
        elif ns.kind in ("instance", "inherited"):
            _translate_instance_namespace(policy, ns, agent)

    return policy


def _wrap_sub_agent_task(fn_obj):
    """Wrap a sub-agent task function to:

    1. Inject session/on_event from context variables (so parent's session
       and event handler propagate to sub-agent calls made from sandbox code).
    2. Convert TaskClarify/TaskFail to EvalError (so the parent sees error
       output rather than a terminal signal).
    """
    from agex.agent.datatypes import TaskClarify, TaskFail
    from agex.eval.error import EvalError

    def wrapper(*args, **kwargs):
        # Inject session and on_event from context if not explicitly provided
        if "session" not in kwargs:
            kwargs["session"] = _current_session.get()
        if "on_event" not in kwargs:
            kwargs["on_event"] = _current_on_event.get()

        try:
            return fn_obj(*args, **kwargs)
        except TaskClarify as e:
            raise EvalError(f"Sub-agent needs clarification: {e.message}") from e
        except TaskFail as e:
            raise EvalError(f"Sub-agent failed: {e.message}") from e

    # Preserve function metadata for sblite's introspection
    wrapper.__name__ = getattr(fn_obj, "__name__", "task")
    wrapper.__doc__ = getattr(fn_obj, "__doc__", None)
    wrapper.__signature__ = getattr(fn_obj, "__signature__", None)
    wrapper.__annotations__ = getattr(fn_obj, "__annotations__", {})
    wrapper.network_access = getattr(fn_obj, "network_access", False)
    return wrapper


def _translate_main_namespace(policy: Policy, ns, agent: "BaseAgent") -> None:
    """Translate the __main__ virtual namespace (registered fns and classes)."""
    import builtins as _builtins

    # Register functions
    for fn_name, fn_obj in ns.fn_objects.items():
        # Skip builtin open — sblite handles it via filesystem interception
        if fn_obj is _builtins.open:
            continue

        # Wrap sub-agent task functions to convert TaskClarify/TaskFail → EvalError
        actual_fn = fn_obj
        if hasattr(fn_obj, "__agex_task_namespace__"):
            actual_fn = _wrap_sub_agent_task(fn_obj)

        spec = ns.fns.get(fn_name)
        host_fs = getattr(spec, "host_fs_access", False) if spec else False
        net = getattr(spec, "network_access", False) if spec else False
        policy.fn(actual_fn, name=fn_name, host_fs_access=host_fs, network_access=net)

    # Register classes
    for cls_name, resolved_cls in ns.classes.items():
        constructable = getattr(resolved_cls, "constructable", True)
        actual_cls = resolved_cls.cls
        host_fs = getattr(actual_cls, "__agex_host_fs_access__", False)
        net = getattr(actual_cls, "__agex_network_access__", False)
        policy.cls(
            actual_cls,
            name=cls_name,
            constructable=constructable,
            host_fs_access=host_fs,
            network_access=net,
        )


def _translate_module_namespace(policy: Policy, ns) -> None:
    """Translate a module namespace."""
    try:
        mod = ns._ensure_module_loaded()
    except Exception:
        return

    configure = _translate_configure(ns.configure) if ns.configure else {}
    policy.module(
        mod,
        name=ns.name,
        include=ns.include,
        exclude=ns.exclude,
        configure=configure,
        recursive=ns.recursive,
        host_fs_access=ns.host_fs_access,
        network_access=ns.network_access,
    )


def _translate_instance_namespace(policy: Policy, ns, agent: "BaseAgent") -> None:
    """Translate an instance namespace (live object)."""
    obj = agent._host_object_registry.get(ns.name)
    if obj is None:
        return

    configure = _translate_configure(ns.configure) if ns.configure else {}
    policy.module(
        obj,
        name=ns.name,
        include=ns.include,
        exclude=ns.exclude,
        configure=configure,
        host_fs_access=ns.host_fs_access,
        network_access=ns.network_access,
    )
