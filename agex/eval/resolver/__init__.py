from typing import TYPE_CHECKING, Any

from agex.eval.error import EvalError
from agex.eval.objects import AgexAttributeError, AgexInstance, AgexModule, AgexObject

from .policy import PolicyFinder
from .vfs import VFSFinder

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.state.core import State


class Resolver:
    """Coordinates module discovery and attribute resolution for the sandbox."""

    def __init__(self, agent: "BaseAgent", session: str = "default"):
        self.agent = agent
        self.session = session

        self.policy_finder = PolicyFinder(agent)
        self.vfs_finder = VFSFinder(agent, session)

        # Initialize finders in order of precedence (Policy first to prevent shadowing)
        self.finders = [self.policy_finder, self.vfs_finder]

    # --- Name Resolution ---
    def resolve_name(self, name: str, state: "State", node) -> Any:
        from agex.eval.builtins import BUILTINS
        from agex.eval.user_errors import AgexNameError

        # 1. Builtins
        if name in BUILTINS:
            return BUILTINS[name]

        # 2. State
        value = state.get(name)
        if value is not None or name in state:
            return value

        # 3. Policy (Live objects, Functions, Classes)
        res = self.policy_finder.resolve_name(name)
        if res is not None:
            return res

        raise AgexNameError(f"name '{name}' is not defined", node)

    # --- Attribute Resolution ---
    def resolve_attribute(self, value: Any, attr_name: str, node) -> Any:
        from agex.eval.objects import AgexVFSModule, BoundInstanceObject
        from agex.eval.utils import get_allowed_attributes_for_instance

        if isinstance(value, (AgexObject, AgexInstance, AgexVFSModule)):
            return value.getattr(attr_name)

        # Host object proxy
        if isinstance(value, BoundInstanceObject):
            return value.getattr(attr_name)

        # AgexModule attribute access with JIT resolution
        if isinstance(value, AgexModule):
            # Check cached submodules first (set via resolve_module)
            try:
                return value.getattr(attr_name)
            except (AgexAttributeError, AttributeError):
                pass

            # First check if this is a registered submodule in policy (static alias)
            parent_ns = self.agent._policy.namespaces.get(value.name)  # type: ignore[attr-defined]
            if (
                parent_ns
                and hasattr(parent_ns, "submodules")
                and attr_name in parent_ns.submodules
            ):
                child_ns_name = parent_ns.submodules[attr_name]
                return AgexModule(
                    name=child_ns_name, agent_fingerprint=self.agent.fingerprint
                )

            # Resolve via Policy logic
            res = self.policy_finder.resolve_attribute(value.name, attr_name)
            if res is not None:
                return res

            raise AgexAttributeError(
                f"module '{value.name}' has no attribute '{attr_name}'", node
            )

        # Handle class attribute access (e.g., datetime.datetime.now)
        target_class = value if isinstance(value, type) else type(value)
        member = self.agent._policy.resolve_class_member(target_class, attr_name)
        if member is not None:
            try:
                return getattr(value, attr_name)
            except AttributeError:
                raise AgexAttributeError(
                    f"'{type(value).__name__}' object has no attribute '{attr_name}'",
                    node,
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
    def resolve_module(self, module_name: str, state: "State", node) -> Any:
        """Resolve a module name to a module object using finders and loaders."""
        # 1. Access or initialize the shared module cache for this session
        cache_key = f"__agex_modules__{self.session}"
        module_cache = state.base_store.get(cache_key)
        if module_cache is None:
            module_cache = {}
            state.base_store.set(cache_key, module_cache)

        # 2. Iteratively resolve segments to handle packages
        parts = module_name.split(".")
        current_full_name = ""
        current_mod = None

        for part in parts:
            if current_full_name:
                current_full_name += "." + part
            else:
                current_full_name = part

            next_mod = None
            spec = None

            # A. Discovery: Ask Finders for a Spec
            for finder in self.finders:
                spec = finder.find_spec(current_full_name)
                if spec:
                    break

            # B. Loading Decision
            if spec:
                # RELOAD POLICY: Always reload VFS modules to pick up changes.
                # Policy modules are essentially singletons and can be cached.
                if spec.origin.startswith("vfs_"):
                    next_mod = spec.loader.load(spec, state)
                elif current_full_name in module_cache:
                    next_mod = module_cache[current_full_name]
                else:
                    next_mod = spec.loader.load(spec, state)
            else:
                # Fallback: check cache for something resolved in a previous task
                if current_full_name in module_cache:
                    next_mod = module_cache[current_full_name]

            if next_mod is None:
                raise EvalError(
                    f"Module '{module_name}' is not registered or whitelisted.",
                    node,
                )

            # Update cache and parent linkage
            module_cache[current_full_name] = next_mod
            state.base_store.set(cache_key, module_cache)

            if current_mod and hasattr(current_mod, "setattr"):
                current_mod.setattr(part, next_mod)

            current_mod = next_mod

        return current_mod

    def import_from(
        self, module_name: str, member_name: str, state: "State", node
    ) -> Any:
        """Handles 'from <module> import <name>'."""
        from types import ModuleType

        # Legacy special-case
        if module_name == "dataclasses":
            raise EvalError(f"No module named '{module_name}' is registered.", node)

        # 1. Check Policy (Whitelist)
        res = self.agent._policy.resolve_module_member(module_name, member_name)
        if res is None:
            # Check if module_name is a dotted path like "os.path" where the submodule
            # is registered separately. Look for parent.submodule relationships.
            if "." in module_name:
                parts = module_name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent_ns = self.agent._policy.namespaces.get(parent_name)  # type: ignore[attr-defined]
                    if parent_ns and hasattr(parent_ns, "submodules"):
                        actual_child_ns_name = parent_ns.submodules.get(child_name)
                        if actual_child_ns_name:
                            # Try resolving from the actual child namespace
                            res = self.agent._policy.resolve_module_member(
                                actual_child_ns_name, member_name
                            )

            # Fallback for recursive parents: allow 'from parent.child import leaf'
            if res is None:
                parent_ns_name = None
                for ns_name, ns in self.agent._policy.namespaces.items():  # type: ignore[attr-defined]
                    if getattr(ns, "kind", None) != "module":
                        continue
                    if not getattr(ns, "recursive", False):
                        continue
                    if module_name.startswith(ns_name + "."):
                        parent_ns_name = ns_name
                        break
                if parent_ns_name is not None:
                    suffix = module_name[len(parent_ns_name) + 1 :]
                    dotted_member = f"{suffix}.{member_name}"
                    res = self.agent._policy.resolve_module_member(
                        parent_ns_name, dotted_member
                    )

        if res is not None:
            # If the resolved member is itself a module, return it wrapped as an
            # AgexModule so that subsequent attribute access goes through policy
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

        # 2. Check VFS / Submodules
        try:
            # A. Try resolving the base module
            # If it's a VFS module, this will load/reload it
            mod = self.resolve_module(module_name, state, node)

            # B. Check if member_name is an attribute/member of the module
            try:
                return self.resolve_attribute(mod, member_name, node)
            except (AgexAttributeError, AttributeError):
                # C. Try resolving as a submodule: from pkg import sub
                full_sub_name = f"{module_name}.{member_name}"
                try:
                    return self.resolve_module(full_sub_name, state, node)
                except EvalError:
                    pass
        except Exception:
            pass

        raise EvalError(f"No name '{member_name}' in module '{module_name}'", node)
