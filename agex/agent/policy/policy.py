from types import ModuleType
from typing import Any, Callable

from ..datatypes import MemberSpec
from ._sync import _sync_submodule_attributes
from .datatypes import (
    RESERVED_NAMES,
    Namespace,
    Pattern,
    ResolutionScope,
    ResolvedClass,
    ResolvedFn,
    ResolvedObj,
    Visibility,
)
from .resolve import (
    _build_registered_class,
    _resolve_class_member,
    make_predicate,
    resolve_class,
    resolve_member,
)


class AgentPolicy:
    """
    Standalone policy engine that manages unified, lazy namespaces and provides
    resolution utilities for modules and classes.
    """

    def __init__(self) -> None:
        self.namespaces: dict[str, Namespace] = {}
        # Keep per-class namespace specs so we can accurately describe classes
        # with their own include/exclude/configure rules during rendering.
        self._class_namespaces: dict[type, Namespace] = {}

    def copy(self) -> "AgentPolicy":
        """
        Create a copy of this policy.

        The copy has independent namespace dictionaries but shares references to
        live objects (modules, functions, classes, instances). This allows the
        copy to be modified (e.g., adding new registrations) without affecting
        the original policy.

        Returns:
            A new AgentPolicy with copied structure but shared live objects.
        """
        from dataclasses import fields

        new_policy = AgentPolicy()

        # Copy each namespace
        for name, ns in self.namespaces.items():
            # Create a new Namespace with the same init fields
            init_kwargs = {
                f.name: getattr(ns, f.name) for f in fields(Namespace) if f.init
            }
            # Copy mutable init fields (dicts) to avoid sharing
            if "configure" in init_kwargs and init_kwargs["configure"]:
                init_kwargs["configure"] = init_kwargs["configure"].copy()
            if "submodules" in init_kwargs and init_kwargs["submodules"]:
                init_kwargs["submodules"] = init_kwargs["submodules"].copy()

            new_ns = Namespace(**init_kwargs)

            # Copy non-init fields (these are set after construction)
            new_ns.fns = ns.fns.copy()
            new_ns.fn_objects = ns.fn_objects.copy()
            new_ns.consts = ns.consts.copy()
            new_ns.classes = ns.classes.copy()

            new_policy.namespaces[name] = new_ns

        # Copy class namespaces (maps type -> Namespace)
        # The Namespace objects here should reference the ones we just copied
        for cls, ns in self._class_namespaces.items():
            if ns.name in new_policy.namespaces:
                new_policy._class_namespaces[cls] = new_policy.namespaces[ns.name]
            else:
                # This namespace isn't in the main dict, copy it separately
                init_kwargs = {
                    f.name: getattr(ns, f.name) for f in fields(Namespace) if f.init
                }
                if "configure" in init_kwargs and init_kwargs["configure"]:
                    init_kwargs["configure"] = init_kwargs["configure"].copy()
                new_ns = Namespace(**init_kwargs)
                new_ns.fns = ns.fns.copy()
                new_ns.fn_objects = ns.fn_objects.copy()
                new_ns.consts = ns.consts.copy()
                new_ns.classes = ns.classes.copy()
                new_policy._class_namespaces[cls] = new_ns

        return new_policy

    # ----- Registration (spec only, no enumeration) -----
    def register_module(
        self,
        *,
        name: str | None = None,
        module: ModuleType | str,
        visibility: Visibility = "medium",
        include: Pattern | None = "*",
        exclude: Pattern | None = ("_*", "*._*"),
        configure: dict[str, MemberSpec] | None = None,
        recursive: bool = False,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> Namespace:
        mod_name = name or (module if isinstance(module, str) else module.__name__)
        spec = Namespace(
            name=mod_name,
            kind="module",
            module=module,
            visibility=visibility,
            include=include,
            exclude=list(exclude) if isinstance(exclude, tuple) else exclude,
            configure=configure or {},
            recursive=recursive,
            host_fs_access=host_fs_access,
            network_access=network_access,
        )
        self.namespaces[mod_name] = spec

        # Populate _class_namespaces for classes in the module
        # This allows method access on instances of these classes
        if not isinstance(module, str):
            import inspect

            from .resolve import make_predicate

            include_pred = make_predicate(include)
            exclude_pred = make_predicate(exclude)

            # Iterate through module members and register classes
            for member_name in dir(module):
                if include_pred(member_name) and not exclude_pred(member_name):
                    member = getattr(module, member_name)
                    if inspect.isclass(member):
                        # Register this class in _class_namespaces so methods are accessible
                        self._class_namespaces[member] = spec

        # Sync submodule attributes so parent.child works when both are registered
        _sync_submodule_attributes(self.namespaces)

        return spec

    def register_instance(
        self,
        *,
        name: str,
        obj: Any,
        visibility: Visibility = "medium",
        include: Pattern | None = "*",
        exclude: Pattern | None = ("_*", "*._*"),
        configure: dict[str, MemberSpec] | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> Namespace:
        spec = Namespace(
            name=name,
            kind="instance",
            obj=obj,
            visibility=visibility,
            include=include,
            exclude=list(exclude) if isinstance(exclude, tuple) else exclude,
            configure=configure or {},
            recursive=False,
            host_fs_access=host_fs_access,
            network_access=network_access,
        )
        self.namespaces[name] = spec
        return spec

    # Virtual main namespace utilities
    def _get_or_create_main(self) -> Namespace:
        ns = self.namespaces.get("__main__")
        if ns is None:
            ns = Namespace(name="__main__", kind="virtual", visibility="high")
            self.namespaces["__main__"] = ns
        return ns

    def register_fn(
        self,
        *,
        func: Callable,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> Namespace:
        final_name = name or getattr(func, "__name__", None) or "fn"
        if final_name in RESERVED_NAMES:
            raise ValueError(
                f"The name '{final_name}' is reserved and cannot be registered."
            )
        main = self._get_or_create_main()
        # Store metadata only; callable binding is host-side concern in this prototype
        final_doc = (
            docstring if docstring is not None else getattr(func, "__doc__", None)
        )
        main.fns[final_name] = MemberSpec(
            visibility=visibility,
            docstring=final_doc,
            host_fs_access=host_fs_access,
            network_access=network_access,
        )
        main.fn_objects[final_name] = func
        return main

    def register_cls(
        self,
        *,
        cls: type,
        name: str | None = None,
        visibility: Visibility = "high",
        constructable: bool = True,
        include: Pattern | None = "*",
        exclude: Pattern | None = "_*",
        configure: dict[str, MemberSpec] | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> Namespace:
        # Build a class spec using a synthetic namespace spec carrying filters
        temp_spec = Namespace(
            name="__main__",
            kind="virtual",
            visibility=visibility,
            include=include,
            exclude=exclude,
            configure=configure or {},
            host_fs_access=host_fs_access,
            network_access=network_access,
        )
        # Respect constructable override via configure at class-level
        cfg = temp_spec.configure.get(cls.__name__, MemberSpec())
        if cfg.constructable is None:
            cfg.constructable = constructable
            temp_spec.configure[cls.__name__] = cfg

        rc = _build_registered_class(cls, temp_spec)
        main = self._get_or_create_main()
        class_key = name or cls.__name__
        main.classes[class_key] = rc
        # Persist the per-class namespace so describe_class can use the
        # correct include/exclude/configure when rendering definitions.
        self._class_namespaces[cls] = temp_spec
        return main

    # ----- Resolution helpers -----
    def resolve_module_member(
        self, namespace: str, member_name: str
    ) -> ResolvedFn | ResolvedClass | ResolvedObj | None:
        spec = self.namespaces.get(namespace)
        if not spec:
            return None
        scope = ResolutionScope(namespaces=self.namespaces)
        return resolve_member(spec, member_name, scope)

    def resolve_class_spec(self, py_cls: type) -> ResolvedClass | None:
        scope = ResolutionScope(namespaces=self.namespaces)
        result = resolve_class(py_cls, None, scope)
        return result if isinstance(result, ResolvedClass) else None

    def resolve_class_member(
        self, py_cls: type, member_name: str
    ) -> ResolvedFn | ResolvedObj | None:
        # Prefer per-class namespace captured at registration for accurate filters
        per_cls_ns = self._class_namespaces.get(py_cls)
        if per_cls_ns is not None:
            # First, try to resolve against actual class members
            res = _resolve_class_member(py_cls, member_name, per_cls_ns)
            if res is not None:
                return res
            # If not found on the class, determine if policy allows this name
            include_pred = make_predicate(per_cls_ns.include)
            exclude_pred = make_predicate(per_cls_ns.exclude)
            dotted_key = f"{py_cls.__name__}.{member_name}"
            forced = dotted_key in per_cls_ns.configure
            allowed = forced or (
                include_pred(member_name) and not exclude_pred(member_name)
            )
            if allowed:
                # Instance-only attribute permitted by policy (e.g., set in __init__ or dataclass field)
                return ResolvedObj(value=None)
            return None
        scope = ResolutionScope(namespaces=self.namespaces)
        result = resolve_class(py_cls, member_name, scope)
        if isinstance(result, (ResolvedFn, ResolvedObj)):
            return result
        return None
