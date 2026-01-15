from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

from agex.agent.policy.resolve import make_predicate

from .builtins import BUILTINS
from .error import EvalError
from .objects import AgexInstance, AgexModule, AgexObject, BoundInstanceObject
from .user_errors import AgexAttributeError, AgexNameError
from .utils import get_allowed_attributes_for_instance

if TYPE_CHECKING:
    from agex.state import State

    from .objects import AgexVFSModule


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

            # Get MemberSpec to extract host_fs_access
            main_ns = self.agent._policy.namespaces.get("__main__")
            member_spec = main_ns.fns.get(name) if main_ns else None
            host_fs_access = (
                getattr(member_spec, "host_fs_access", False) if member_spec else False
            )

            return NativeFunction(name=name, fn=res.fn, host_fs_access=host_fs_access)  # type: ignore[attr-defined]

        # 5. Registered classes via policy
        res = self.agent._policy.resolve_module_member("__main__", name)
        if res is not None and hasattr(res, "cls"):
            return res.cls  # type: ignore[attr-defined]

        raise AgexNameError(f"name '{name}' is not defined", node)

    # --- Attribute Resolution ---
    def resolve_attribute(self, value: Any, attr_name: str, node) -> Any:
        # Sandboxed AgexObjects, VFS modules and live objects have their own logic
        from .objects import AgexVFSModule

        if isinstance(value, (AgexObject, AgexInstance, AgexVFSModule)):
            return value.getattr(attr_name)

        # Host object proxy
        if isinstance(value, BoundInstanceObject):
            return value.getattr(attr_name)

        # AgexModule attribute access with JIT resolution
        if isinstance(value, AgexModule):
            # First check if this is a registered submodule
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

        # Handle class attribute access (e.g., datetime.datetime.now)
        # If value is a class, resolve its members directly, not the type's members
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
    def resolve_module(self, module_name: str, state: State, node) -> Any:
        # 1. Check Policy (Whitelist)
        # Use policy.resolve_module to check if the module name itself is registered
        if module_name in self.agent._policy.namespaces:
            return AgexModule(
                name=module_name, agent_fingerprint=self.agent.fingerprint
            )

        # For recursive modules, check if any registered namespace is a parent
        for ns_name, ns in self.agent._policy.namespaces.items():  # type: ignore[attr-defined]
            if getattr(ns, "recursive", False) and module_name.startswith(
                ns_name + "."
            ):
                return AgexModule(
                    name=module_name, agent_fingerprint=self.agent.fingerprint
                )

        # 2. Check VFS

        try:
            # Check for module file in VFS
            filename = f"{module_name}.py"
            fs = self.agent.fs()
            if fs.exists(filename):
                return self._load_vfs_module(module_name, state, node)
        except Exception:
            # Filesystem error or not configured - fall through
            pass

        raise EvalError(
            f"Module '{module_name}' is not registered or whitelisted.", node
        )

    def _load_vfs_module(self, name: str, state: State, node) -> AgexVFSModule:
        """Load and execute a module from the VFS into a namespaced state."""
        from agex.state import Namespaced

        from .core import evaluate_program
        from .objects import AgexVFSModule

        # Get the underlying base store for namespacing
        base = state.base_store

        # Create isolated namespaced state: modules/<name>
        # Nested Namespaced to avoid forbidden slashes in segment names
        root_ns = Namespaced(base, "modules")
        module_state = Namespaced(root_ns, name)

        # NOTE: AgexVFSModules act as modules, so they should be recursive
        # But `Namespaced` doesn't have a `recursive` attribute.
        # The `AgexModule` wrapper returned by `resolve_module` handles dotted paths IF
        # the policy namespace has `recursive=True`.
        # Here we are returning `AgexVFSModule`.

        # CLEAR OLD STATE: ensure clean slate for re-loading/overwriting
        # descendant_keys() gets all keys including those in deeper sub-namespaces
        for key in list(module_state.descendant_keys()):
            module_state.remove(key)

        # Read and parse code
        filename = f"{name}.py"
        try:
            fs = self.agent.fs()
            code_bytes = fs.read(filename)
            code = code_bytes.decode("utf-8")
        except Exception as e:
            raise EvalError(f"Failed to read module '{name}' from VFS: {e}", node)

        # Execute module code into its namespace
        try:
            evaluate_program(
                code,
                self.agent,
                state=module_state,
                # Ensure it has access to the same main loop/callbacks if needed
                # (Evaluator might need to pass these through to Resolver)
            )
        except Exception as e:
            from agex.agent.datatypes import _AgentExit

            if isinstance(e, _AgentExit):
                raise
            raise EvalError(f"Error initializing module '{name}': {e}", node) from e

        return AgexVFSModule(
            name=name, state=module_state, agent_fingerprint=self.agent.fingerprint
        )

    def import_from(
        self, module_name: str, member_name: str, state: State, node
    ) -> Any:
        # Preserve legacy special-case: only allow `from dataclasses import dataclass` as a no-op.
        # For any other import from dataclasses, treat module as unregistered.
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

        # 2. Check VFS
        try:
            # Check if this is a VFS module
            filename = f"{module_name}.py"
            fs = self.agent.fs()
            if fs.exists(filename):
                # Load module to populate namespaced state
                vfs_mod = self._load_vfs_module(module_name, state, node)
                return vfs_mod.getattr(member_name)
        except Exception:
            pass

        raise EvalError(
            f"Cannot import name '{member_name}' from module '{module_name}'.",
            node,
        )
