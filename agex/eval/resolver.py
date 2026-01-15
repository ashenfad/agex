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

    def __init__(self, agent, session: str = "default"):
        self.agent = agent
        self.session = session
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
            # Check cached submodules first (set via resolve_module)
            try:
                return value.getattr(attr_name)
            except (AgexAttributeError, AttributeError):
                pass

            # First check if this is a registered submodule in policy
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
        # 1. Access or initialize the shared module cache for this session
        cache_key = f"__agex_modules__{self.session}"
        module_cache = state.base_store.get(cache_key)
        if module_cache is None:
            module_cache = {}
            state.base_store.set(cache_key, module_cache)

        # 2. Split name and resolve iteratively to handle packages/submodules
        parts = module_name.split(".")
        current_full_name = ""
        current_mod = None

        for part in parts:
            if current_full_name:
                current_full_name += "." + part
            else:
                current_full_name = part

            next_mod = None

            # A. Check Policy FIRST (Prevent Shadowing)
            if current_full_name in self.agent._policy.namespaces:
                next_mod = AgexModule(
                    name=current_full_name, agent_fingerprint=self.agent.fingerprint
                )
            else:
                # Check for recursive policy parents
                for ns_name, ns in self.agent._policy.namespaces.items():  # type: ignore[attr-defined]
                    if getattr(ns, "recursive", False) and current_full_name.startswith(
                        ns_name + "."
                    ):
                        next_mod = AgexModule(
                            name=current_full_name,
                            agent_fingerprint=self.agent.fingerprint,
                        )
                        break

                # Check for child policy matches (implicit parents)
                if next_mod is None:
                    for ns_name in self.agent._policy.namespaces:  # type: ignore[attr-defined]
                        if ns_name.startswith(current_full_name + "."):
                            next_mod = AgexModule(
                                name=current_full_name,
                                agent_fingerprint=self.agent.fingerprint,
                            )
                            break

            # B. Check VFS (Reload Support)
            # If not in policy, check if file exists in VFS
            if next_mod is None:
                current_path = current_full_name.replace(".", "/")
                try:
                    if self.agent._fs_exists(f"{current_path}.py", self.session):
                        next_mod = self._load_vfs_module(current_full_name, state, node)
                    elif self.agent._fs_exists(
                        f"{current_path}/__init__.py", self.session
                    ):
                        next_mod = self._load_vfs_module(current_full_name, state, node)
                    elif self.agent._fs_exists(f"{current_path}/", self.session):
                        # Namespace package
                        next_mod = self._load_vfs_module(current_full_name, state, node)
                except Exception:
                    pass

            # C. Check Cache (Fallback for cached policy modules)
            if next_mod is None and current_full_name in module_cache:
                next_mod = module_cache[current_full_name]

            if next_mod is None:
                # Restore original error message for compatibility
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

        # CLEAR OLD STATE: ensure clean slate for re-loading/overwriting
        # We must preserve sub-namespaces (submodules) during reload!
        # Only remove non-namespaced keys.
        from agex.state import Namespaced

        for key in list(module_state.keys()):
            val = module_state.get(key)
            if not isinstance(val, (AgexModule, AgexVFSModule)):
                module_state.remove(key)

        # Map dotted name to directory/file path: pkg.sub -> pkg/sub
        path_prefix = name.replace(".", "/")

        # Discovery and Loading Strategy
        code = None
        target_path = None

        # 1. Try Package (__init__.py)
        init_path = f"{path_prefix}/__init__.py"
        if self.agent._fs_exists(init_path, self.session):
            target_path = init_path
        else:
            # 2. Try Module (.py)
            py_path = f"{path_prefix}.py"
            if self.agent._fs_exists(py_path, self.session):
                target_path = py_path

        if target_path:
            try:
                code_bytes = self.agent._fs_read(target_path, self.session)
                code = code_bytes.decode("utf-8")
            except Exception as e:
                raise EvalError(f"Failed to read module '{name}' from VFS: {e}", node)

        # 3. Handle Namespace Package (dir exists but no code yet)
        if code is None:
            if not self.agent._fs_exists(f"{path_prefix}/", self.session):
                raise EvalError(f"Module '{name}' not found in VFS", node)
            # Namespace package has no top-level code, but state is initialized empty.
            code = ""

        # Execute module code into its namespace
        try:
            # Set standard module attributes
            module_state.set("__name__", name)
            module_state.set("__file__", target_path or f"<virtual:{name}>")
            module_state.set(
                "__package__", name.rsplit(".", 1)[0] if "." in name else ""
            )

            evaluate_program(
                code,
                self.agent,
                state=module_state,
                session=self.session,
            )
        except Exception as e:
            from agex.agent.datatypes import _AgentExit

            if isinstance(e, _AgentExit):
                raise
            raise EvalError(f"Error initializing module '{name}': {e}", node) from e

        return AgexVFSModule(
            name=name,
            state=module_state,
            agent_fingerprint=self.agent.fingerprint,
            session=self.session,
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
            # A. Try loading the base module
            vfs_mod = self.resolve_module(module_name, state, node)

            # B. Check if member_name is already in the module (function, class, etc)
            try:
                return vfs_mod.getattr(member_name)
            except (AgexAttributeError, AttributeError):
                # C. Not a member, try resolving as a submodule: from pkg import sub
                full_sub_name = f"{module_name}.{member_name}"
                try:
                    return self.resolve_module(full_sub_name, state, node)
                except EvalError:
                    # Not a submodule either
                    pass
        except Exception:
            pass

        raise EvalError(
            f"Cannot import name '{member_name}' from module '{module_name}'.",
            node,
        )
