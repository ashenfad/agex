from __future__ import annotations

from types import ModuleType
from typing import Any

from agex.agent.policy.resolve import make_predicate

from .builtins import BUILTINS
from .error import EvalError
from .objects import AgexInstance, AgexModule, AgexObject, BoundInstanceObject
from .user_errors import AgexAttributeError, AgexNameError
from .utils import get_allowed_attributes_for_instance


class Resolver:
    """
    Resolve policies to discover whether artifacts are whitelisted.
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

        raise AgexNameError(f"name '{name}' is not defined", node)

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
            # Prefer exact namespace match
            res = self.agent._policy.resolve_module_member(value.name, attr_name)
            # If no exact match, try resolving against the nearest registered parent namespace
            if res is None and "." in value.name:
                # Find longest namespace that is a prefix of value.name
                parent_ns_name = None
                for ns_name in sorted(
                    self.agent._policy.namespaces.keys(), key=len, reverse=True
                ):  # type: ignore[attr-defined]
                    if value.name.startswith(ns_name + "."):
                        parent_ns_name = ns_name
                        break
                if parent_ns_name is not None:
                    # Compose a dotted member path relative to the parent module
                    suffix = value.name[len(parent_ns_name) + 1 :]
                    dotted_member = suffix + "." + attr_name
                    res = self.agent._policy.resolve_module_member(
                        parent_ns_name, dotted_member
                    )
            if res is None:
                # Fallback: if a child submodule is registered as its own namespace, return it
                # Compute the fully qualified module path for the child attribute
                try:
                    parent_spec = self.agent._policy.namespaces.get(value.name)  # type: ignore[attr-defined]
                except Exception:
                    parent_spec = None
                if (
                    parent_spec is not None
                    and getattr(parent_spec, "kind", None) == "module"
                ):
                    try:
                        parent_mod = parent_spec._ensure_module_loaded()
                        dotted = f"{parent_mod.__name__}.{attr_name}"
                        # Find a registered namespace matching this dotted module path
                        for ns_name, ns in self.agent._policy.namespaces.items():  # type: ignore[attr-defined]
                            if getattr(ns, "kind", None) != "module":
                                continue
                            loaded = None
                            try:
                                loaded = ns._ensure_module_loaded()
                            except Exception:
                                continue
                            if (
                                isinstance(loaded, ModuleType)
                                and loaded.__name__ == dotted
                            ):
                                return AgexModule(
                                    name=ns_name,
                                    agent_fingerprint=self.agent.fingerprint,
                                )
                    except Exception:
                        pass
                raise AgexAttributeError(
                    f"module '{value.name}' has no attribute '{attr_name}'", node
                )
            # If the resolved member is a submodule, prefer an existing registered
            # namespace for that module (supports independent submodule registration).
            submod = getattr(res, "value", None)
            if isinstance(submod, ModuleType):
                # Look for a namespace whose loaded module object matches this submodule
                for ns_name, ns in getattr(self.agent._policy, "namespaces").items():  # type: ignore[attr-defined]
                    if getattr(ns, "kind", None) == "module":
                        try:
                            loaded = ns._ensure_module_loaded()
                        except Exception:
                            continue
                        if loaded is submod:
                            return AgexModule(
                                name=ns_name,
                                agent_fingerprint=self.agent.fingerprint,
                            )
                # Otherwise, wrap as a dotted child of the current module name
                return AgexModule(
                    name=f"{value.name}.{attr_name}",
                    agent_fingerprint=self.agent.fingerprint,
                )
            return getattr(res, "fn", None) or getattr(res, "cls", None) or submod

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
        # If the resolved member is itself a module, return it wrapped as an
        # AgexModule so that subsequent attribute access goes through policy
        # (enabling constants and dotted resolution with include/exclude gating).
        val = (
            getattr(res, "fn", None)
            or getattr(res, "cls", None)
            or getattr(res, "value", None)
        )
        if isinstance(val, ModuleType):
            # Prefer an existing registered namespace for the resolved module
            for ns_name, ns in getattr(self.agent._policy, "namespaces").items():  # type: ignore[attr-defined]
                if getattr(ns, "kind", None) != "module":
                    continue
                try:
                    loaded = ns._ensure_module_loaded()
                except Exception:
                    continue
                if loaded is val:
                    return AgexModule(
                        name=ns_name, agent_fingerprint=self.agent.fingerprint
                    )
            # Otherwise compose a dotted child path relative to the parent module name
            dotted_name = f"{module_name}.{member_name}"
            return AgexModule(
                name=dotted_name, agent_fingerprint=self.agent.fingerprint
            )
        return val
