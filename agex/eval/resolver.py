from __future__ import annotations

from typing import Any

from agex.agent.policy.resolve import make_predicate

from .builtins import BUILTINS
from .error import EvalError
from .objects import AgexInstance, AgexModule, AgexObject, BoundInstanceObject
from .user_errors import AgexAttributeError
from .utils import get_allowed_attributes_for_instance


class Resolver:
    """
    Facade over the agent's eager registries used by the evaluator.

    This centralizes name, attribute, and import resolution so we can later
    swap it with a lazy policy-based implementation without touching evaluator
    logic.
    """

    def __init__(self, agent):
        self.agent = agent
        # Policy-backed resolution only

    # --- Name Resolution ---
    def resolve_name(self, name: str, state, node) -> Any:
        # 1. Builtins
        if name in BUILTINS:
            return BUILTINS[name]

        # 2. State
        value = state.get(name)
        if value is not None or name in state:
            return value

        # 3. Registered live objects via policy instance namespaces
        ns = self.agent._policy.namespaces.get(name)  # type: ignore[attr-defined]
        if ns is not None and getattr(ns, "kind", None) == "instance":
            from agex.agent.datatypes import MemberSpec, RegisteredObject

            from .objects import BoundInstanceObject

            methods: dict[str, MemberSpec] = {}
            properties: dict[str, MemberSpec] = {}
            live_obj = self.agent._host_object_registry.get(name)
            if live_obj is not None:
                include_pred = make_predicate(ns.include)
                exclude_pred = make_predicate(ns.exclude)
                for attr in dir(live_obj):
                    if attr.startswith("@"):
                        continue
                    if not (include_pred(attr) and not exclude_pred(attr)):
                        continue
                    try:
                        value = getattr(live_obj, attr)
                    except Exception:
                        continue
                    cfg = ns.configure.get(attr, MemberSpec())
                    vis = cfg.visibility or ns.visibility
                    doc = cfg.docstring
                    if callable(value):
                        methods[attr] = MemberSpec(visibility=vis, docstring=doc)
                    else:
                        properties[attr] = MemberSpec(visibility=vis, docstring=doc)
            else:
                # Fallback to configured names only if live object missing
                for attr, cfg in ns.configure.items():
                    if attr.startswith("__"):
                        continue
                    vis = cfg.visibility or ns.visibility
                    methods[attr] = MemberSpec(visibility=vis, docstring=cfg.docstring)

            reg_object = RegisteredObject(
                name=name,
                visibility=ns.visibility,
                methods=methods,
                properties=properties,
                exception_mappings=getattr(ns, "exception_mappings", {}),
            )
            return BoundInstanceObject(
                reg_object=reg_object, host_registry=self.agent._host_object_registry
            )

        # 4. Registered functions via policy
        res = self.agent._policy.resolve_module_member("__main__", name)
        if res is not None and hasattr(res, "fn"):
            from .functions import NativeFunction

            return NativeFunction(name=name, fn=res.fn)  # type: ignore[attr-defined]

        # 5. Registered classes via policy
        res = self.agent._policy.resolve_module_member("__main__", name)
        if res is not None and hasattr(res, "cls"):
            return res.cls  # type: ignore[attr-defined]

        raise EvalError(f"Name '{name}' is not defined. (forgot import?)", node)

    # --- Attribute Resolution ---
    def resolve_attribute(self, value: Any, attr_name: str, node) -> Any:
        # Sandboxed AgexObjects and live objects have their own logic
        if isinstance(value, (AgexObject, AgexInstance)):
            return value.getattr(attr_name)

        # Host object proxy
        if isinstance(value, BoundInstanceObject):
            return value.getattr(attr_name)

        # AgexModule attribute access with JIT resolution
        if isinstance(value, AgexModule):
            res = self.agent._policy.resolve_module_member(value.name, attr_name)
            if res is None:
                raise AgexAttributeError(
                    f"module '{value.name}' has no attribute '{attr_name}'", node
                )
            return (
                getattr(res, "fn", None)
                or getattr(res, "cls", None)
                or getattr(res, "value", None)
            )

        # Check for registered host classes and whitelisted methods on Python objects
        allowed_attrs = get_allowed_attributes_for_instance(self.agent, value)
        if attr_name in allowed_attrs:
            try:
                return getattr(value, attr_name)
            except AttributeError:
                raise AgexAttributeError(
                    f"'{type(value).__name__}' object has no attribute '{attr_name}'",
                    node,
                )

        raise AgexAttributeError(
            f"'{type(value).__name__}' object has no attribute '{attr_name}'", node
        )

    # --- Import Resolution ---
    def resolve_module(self, module_name: str, node) -> AgexModule:
        # Creating AgexModule is safe as a capability token; members resolve lazily via policy
        if module_name in self.agent._policy.namespaces:  # type: ignore[attr-defined]
            return AgexModule(
                name=module_name, agent_fingerprint=self.agent.fingerprint
            )
        raise EvalError(
            f"Module '{module_name}' is not registered or whitelisted.", node
        )

    def import_from(self, module_name: str, member_name: str, node) -> Any:
        # Preserve legacy special-case: only allow `from dataclasses import dataclass` as a no-op.
        # For any other import from dataclasses, treat module as unregistered.
        if module_name == "dataclasses":
            raise EvalError(f"No module named '{module_name}' is registered.", node)

        res = self.agent._policy.resolve_module_member(module_name, member_name)
        if res is None:
            raise EvalError(
                f"Cannot import name '{member_name}' from module '{module_name}'.",
                node,
            )
        return (
            getattr(res, "fn", None)
            or getattr(res, "cls", None)
            or getattr(res, "value", None)
        )
