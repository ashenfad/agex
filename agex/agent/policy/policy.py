from types import ModuleType
from typing import Any, Callable

from ..datatypes import MemberSpec
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
        )
        self.namespaces[mod_name] = spec
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
        exception_mappings: dict[type, type] | None = None,
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
        )
        spec.exception_mappings = exception_mappings or {}
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
        main.fns[final_name] = MemberSpec(visibility=visibility, docstring=final_doc)
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
    ) -> Namespace:
        # Build a class spec using a synthetic namespace spec carrying filters
        temp_spec = Namespace(
            name="__main__",
            kind="virtual",
            visibility=visibility,
            include=include,
            exclude=exclude,
            configure=configure or {},
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
