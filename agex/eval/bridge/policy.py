"""
Translates agex's AgentPolicy into a sandtrap Policy.

Maps agex Namespace registrations (module, instance, virtual/__main__)
to sandtrap's policy.fn(), policy.cls(), and policy.module() calls.
"""

from __future__ import annotations

import contextvars
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Callable

from sandtrap.policy import MemberSpec as SandtrapMemberSpec
from sandtrap.policy import Policy

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.agent.datatypes import MemberSpec as AgexMemberSpec

# Context variables for propagating session/on_event/on_token to sub-agent task calls.
# Set by execute_sandboxed/aexecute_sandboxed before running sandbox code.
_current_session: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_session", default="default"
)
_current_on_event: contextvars.ContextVar[Callable[[Any], None] | None] = (
    contextvars.ContextVar("_current_on_event", default=None)
)
_current_on_token: contextvars.ContextVar[Callable[[Any], None] | None] = (
    contextvars.ContextVar("_current_on_token", default=None)
)
# Holds (state, agent_name) for the currently executing sandbox's agent.
# When a sub-task is called from inside sandbox code, its sync_task_func
# reads this to know where to inject synthetic OutputEvents that carry
# the sub-agent's REPORTs into the parent's observation history.
# None at the top level (no parent sandbox active).
_current_parent_log: contextvars.ContextVar[
    tuple[MutableMapping[str, Any], str] | None
] = contextvars.ContextVar("_current_parent_log", default=None)


def _translate_configure(
    agex_configure: dict[str, "AgexMemberSpec"],
) -> dict[str, SandtrapMemberSpec]:
    """Convert agex MemberSpec configure dict to sandtrap MemberSpec configure dict.

    Drops agex-only fields (visibility, docstring, constructable) and keeps
    the security-relevant fields (host_fs_access, network_access).
    """
    result: dict[str, SandtrapMemberSpec] = {}
    for name, spec in agex_configure.items():
        result[name] = SandtrapMemberSpec(
            host_fs_access=getattr(spec, "host_fs_access", False),
            network_access=getattr(spec, "network_access", False),
        )
    return result


def translate_policy(
    agent: "BaseAgent",
    timeout: float | None = None,
    tick_limit: int | None = None,
) -> Policy:
    """Convert an agent's policy registrations into a sandtrap Policy.

    Args:
        agent: The agent whose policy to translate.
        timeout: Execution timeout in seconds. Defaults to agent.eval_timeout_seconds.
        tick_limit: Max checkpoint ticks. Defaults to agent.eval_tick_limit.

    Returns:
        A sandtrap Policy with equivalent registrations.
    """
    effective_timeout = timeout if timeout is not None else agent.eval_timeout_seconds
    effective_tick_limit = (
        tick_limit
        if tick_limit is not None
        else getattr(agent, "eval_tick_limit", None)
    )
    effective_memory_limit = getattr(agent, "max_memory_mb", None)
    policy = Policy(
        timeout=effective_timeout,
        tick_limit=effective_tick_limit,
        memory_limit=effective_memory_limit,
    )

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
    3. Always call the sync task loop so LLM-generated sandbox code doesn't
       need ``await`` and there are no event-loop conflicts.
    """
    from agex.agent.datatypes import TaskClarify, TaskFail
    from agex.eval.error import EvalError

    # Use the sync task func to avoid event-loop conflicts when the
    # orchestrator's sandbox runs inside aexec.
    call_fn = getattr(fn_obj, "_sync_task_func", None) or fn_obj

    def wrapper(*args, **kwargs):
        # Inject session, on_event, on_token from context if not explicitly provided
        if "session" not in kwargs:
            kwargs["session"] = _current_session.get()
        if "on_event" not in kwargs:
            kwargs["on_event"] = _current_on_event.get()
        if "on_token" not in kwargs:
            on_token = _current_on_token.get()
            if on_token is not None:
                kwargs["on_token"] = on_token

        try:
            return call_fn(*args, **kwargs)
        except TaskClarify as e:
            raise EvalError(f"Sub-agent needs clarification: {e.message}") from e
        except TaskFail as e:
            raise EvalError(f"Sub-agent failed: {e.message}") from e

    # Preserve function metadata for sandtrap's introspection
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
        # Skip builtin open — sandtrap handles it via filesystem interception
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
