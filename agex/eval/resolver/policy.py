from typing import TYPE_CHECKING, Any

from kvit import Store

from .base import BaseFinder, BaseLoader, ModuleSpec

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent


class PolicyLoader(BaseLoader):
    """Loader for modules whitelisted in the agent policy."""

    def __init__(self, agent: "BaseAgent"):
        self.agent = agent

    def load(self, spec: ModuleSpec, state: Store) -> Any:
        from agex.eval.objects import AgexModule

        return AgexModule(name=spec.name, agent_fingerprint=self.agent.fingerprint)


class PolicyFinder(BaseFinder):
    """Finder that checks the Agent's configured policy (whitelist)."""

    def __init__(self, agent: "BaseAgent"):
        self.agent = agent
        self.loader = PolicyLoader(agent)

    def find_spec(self, fullname: str) -> ModuleSpec | None:
        # A. Direct Whitelist Match
        if fullname in self.agent._policy.namespaces:
            return ModuleSpec(name=fullname, origin="policy", loader=self.loader)

        # B. Check for recursive policy parents (e.g. 'os.path' if 'os' is recursive)
        for ns_name, ns in self.agent._policy.namespaces.items():  # type: ignore[attr-defined]
            if getattr(ns, "recursive", False) and fullname.startswith(ns_name + "."):
                # Verify the submodule actually exists on the host module
                # to prevent creating placeholders for non-existent imports
                try:
                    parent_mod = ns._ensure_module_loaded()
                    suffix = fullname[len(ns_name) + 1 :]
                    obj = parent_mod
                    for part in suffix.split("."):
                        obj = getattr(obj, part)
                    # Submodule exists - return the spec
                    return ModuleSpec(
                        name=fullname, origin="policy", loader=self.loader
                    )
                except (AttributeError, Exception):
                    # Submodule doesn't exist - don't return a spec
                    pass

        # C. Check for child policy matches (Implicit Parents)
        # If 'numpy.random' is registered but 'numpy' isn't, 'import numpy' must work.
        for ns_name in self.agent._policy.namespaces:  # type: ignore[attr-defined]
            if ns_name.startswith(fullname + "."):
                return ModuleSpec(name=fullname, origin="policy", loader=self.loader)

        return None

    def resolve_name(self, name: str) -> Any:
        """Resolve a name via policy (live objects, functions, classes)."""
        # 1. Registered live objects via policy instance namespaces
        ns = self.agent._policy.namespaces.get(name)  # type: ignore[attr-defined]
        if ns is not None and getattr(ns, "kind", None) == "instance":
            from agex.agent.datatypes import MemberSpec, RegisteredObject
            from agex.agent.policy.resolve import make_predicate
            from agex.eval.objects import BoundInstanceObject

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

        # 2. Registered functions via policy
        res = self.agent._policy.resolve_module_member("__main__", name)
        if res is not None and hasattr(res, "fn"):
            from agex.eval.functions import NativeFunction

            # Get MemberSpec to extract host_fs_access and network_access
            main_ns = self.agent._policy.namespaces.get("__main__")
            member_spec = main_ns.fns.get(name) if main_ns else None
            host_fs_access = (
                getattr(member_spec, "host_fs_access", False) if member_spec else False
            )
            network_access = (
                getattr(member_spec, "network_access", False) if member_spec else False
            )

            return NativeFunction(
                name=name,
                fn=res.fn,
                host_fs_access=host_fs_access,
                network_access=network_access,
            )  # type: ignore[attr-defined]

        # 3. Registered classes via policy
        res = self.agent._policy.resolve_module_member("__main__", name)
        if res is not None and hasattr(res, "cls"):
            return res.cls  # type: ignore[attr-defined]

        return None

    def resolve_attribute(self, module_name: str, attr_name: str) -> Any:
        """Resolve an attribute of a policy module."""
        from types import ModuleType

        from agex.eval.objects import AgexModule

        # Prefer exact namespace match
        res = self.agent._policy.resolve_module_member(module_name, attr_name)

        # If no exact match, try resolving against the nearest registered parent namespace
        if res is None and "." in module_name:
            # Find longest namespace that is a prefix of module_name
            parent_ns_name = None
            for ns_name in sorted(
                self.agent._policy.namespaces.keys(), key=len, reverse=True
            ):  # type: ignore[attr-defined]
                if module_name.startswith(ns_name + "."):
                    parent_ns_name = ns_name
                    break
            if parent_ns_name is not None:
                # Compose a dotted member path relative to the parent module
                suffix = module_name[len(parent_ns_name) + 1 :]
                dotted_member = suffix + "." + attr_name
                res = self.agent._policy.resolve_module_member(
                    parent_ns_name, dotted_member
                )

        if res is None:
            # Fallback: if a child submodule is registered as its own namespace, return it
            try:
                parent_spec = self.agent._policy.namespaces.get(module_name)  # type: ignore[attr-defined]
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
                        if isinstance(loaded, ModuleType) and loaded.__name__ == dotted:
                            return AgexModule(
                                name=ns_name,
                                agent_fingerprint=self.agent.fingerprint,
                            )
                except Exception:
                    pass
            return None

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
                name=f"{module_name}.{attr_name}",
                agent_fingerprint=self.agent.fingerprint,
            )

        return getattr(res, "fn", None) or getattr(res, "cls", None) or submod
