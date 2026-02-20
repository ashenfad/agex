"""
Translates agex's AgentPolicy into a sblite Policy.

Maps agex Namespace registrations (module, instance, virtual/__main__)
to sblite's policy.fn(), policy.cls(), and policy.module() calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sblite.policy import MemberSpec as SbliteMemberSpec
from sblite.policy import Policy

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.agent.datatypes import MemberSpec as AgexMemberSpec


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


def _translate_main_namespace(policy: Policy, ns, agent: "BaseAgent") -> None:
    """Translate the __main__ virtual namespace (registered fns and classes)."""
    # Register functions
    for fn_name, fn_obj in ns.fn_objects.items():
        spec = ns.fns.get(fn_name)
        host_fs = getattr(spec, "host_fs_access", False) if spec else False
        net = getattr(spec, "network_access", False) if spec else False
        policy.fn(fn_obj, name=fn_name, host_fs_access=host_fs, network_access=net)

    # Register classes
    for cls_name, resolved_cls in ns.classes.items():
        constructable = getattr(resolved_cls, "constructable", True)
        policy.cls(
            resolved_cls.cls,
            name=cls_name,
            constructable=constructable,
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
