import importlib
import inspect
from types import ModuleType
from typing import Callable, Iterable

from ..datatypes import MemberSpec
from .datatypes import (
    Namespace,
    Pattern,
    ResolutionScope,
    ResolvedClass,
    ResolvedFn,
    ResolvedObj,
)


def make_predicate(pattern: Pattern | None) -> Callable[[str], bool]:
    if pattern is None:
        return lambda _name: False
    if isinstance(pattern, str):
        import fnmatch

        return lambda name: fnmatch.fnmatch(name, pattern)
    if isinstance(pattern, Iterable):
        sub_preds = [make_predicate(p) for p in pattern]

        def any_match(name: str) -> bool:
            return any(p(name) for p in sub_preds)

        return any_match
    raise TypeError(f"Invalid pattern type: {type(pattern)}")


def _class_allowed_by_namespace(py_cls: type, spec: Namespace) -> bool:
    include_pred = make_predicate(spec.include)
    exclude_pred = make_predicate(spec.exclude)
    cls_name = getattr(py_cls, "__name__", "")
    if cls_name in spec.configure:
        return True
    if any(k.startswith(f"{cls_name}.") for k in spec.configure.keys()):
        return True
    return include_pred(cls_name) and not exclude_pred(cls_name)


def _build_registered_class(py_cls: type, spec: Namespace) -> ResolvedClass:
    cls_cfg = spec.configure.get(py_cls.__name__, MemberSpec())
    constructable = cls_cfg.constructable is None or bool(cls_cfg.constructable)
    return ResolvedClass(cls=py_cls, constructable=constructable)


def _resolve_class_member(
    py_cls: type, member_name: str, spec: Namespace
) -> ResolvedFn | ResolvedObj | None:
    include_pred = make_predicate(spec.include)
    exclude_pred = make_predicate(spec.exclude)
    dot_key = f"{py_cls.__name__}.{member_name}"
    forced = dot_key in spec.configure
    allowed = forced or (include_pred(member_name) and not exclude_pred(member_name))
    if not allowed:
        return None
    member = getattr(py_cls, member_name, None)
    if member is None:
        return None
    if inspect.isroutine(member):
        return ResolvedFn(fn=member)
    else:
        return ResolvedObj(value=member)


def _is_constructable(py_cls: type, spec: Namespace) -> bool:
    cfg = spec.configure.get(py_cls.__name__, MemberSpec())
    return True if cfg.constructable is None else bool(cfg.constructable)


def _resolve_virtual_member(
    spec: Namespace,
    member_name: str,
    include_pred,
    exclude_pred,
    scope: ResolutionScope,
) -> ResolvedFn | ResolvedClass | ResolvedObj | None:
    if "." in member_name:
        cls_name, short = member_name.split(".", 1)
        rc = spec.classes.get(cls_name)
        if rc:
            return _resolve_class_member(rc.cls, short, spec)
        return None
    if member_name in spec.fns:
        fn = spec.fn_objects.get(member_name)
        if fn is None:
            return None
        return ResolvedFn(fn=fn)
    if member_name in spec.classes:
        # Return the registered class directly for virtual namespace
        return spec.classes[member_name]
    if member_name in spec.consts:
        return None
    return None


def _resolve_instance_member(
    spec: Namespace,
    member_name: str,
    include_pred,
    exclude_pred,
    scope: ResolutionScope,
) -> ResolvedFn | ResolvedClass | ResolvedObj | None:
    if not spec.obj:
        return None
    if not (include_pred(member_name) and not exclude_pred(member_name)):
        return None
    member = getattr(spec.obj, member_name, None)
    if member is None:
        return None
    if callable(member):
        return ResolvedFn(fn=member)
    else:
        return ResolvedObj(value=member)


def _resolve_module_member(
    spec: Namespace,
    member_name: str,
    include_pred,
    exclude_pred,
    scope: ResolutionScope,
) -> ResolvedFn | ResolvedClass | ResolvedObj | None:
    mod = spec._ensure_module_loaded()
    # Support recursive resolution for submodule imports via
    #   from pkg import submodule
    # When recursive is enabled and the top-level package does not have a direct
    # attribute named `member_name`, attempt to import `pkg.member_name` as a
    # submodule, respecting exclude patterns.
    if "." not in member_name and spec.recursive:
        # If the package already has a direct attribute, fall through to normal
        # handling below to preserve class/member resolution and include gating.
        if not hasattr(mod, member_name):
            submod_path = f"{mod.__name__}.{member_name}"
            # Use the dotted path for exclude checks
            if not exclude_pred(submod_path):
                try:
                    submod = importlib.import_module(submod_path)
                except Exception:
                    submod = None
                if submod is not None:
                    return ResolvedObj(value=submod)
    if "." in member_name:
        parts = member_name.split(".")
        if len(parts) == 2:
            cls_name, short = parts
            cls_obj = getattr(mod, cls_name, None)
            if inspect.isclass(cls_obj):
                if not _class_allowed_by_namespace(cls_obj, spec):
                    return None
                return _resolve_class_member(cls_obj, short, spec)
        if not spec.recursive:
            return None
        submod_path = f"{mod.__name__}." + ".".join(parts[:-1])
        if exclude_pred(submod_path):
            return None
        try:
            submod = importlib.import_module(submod_path)
        except Exception:
            return None
        leaf = parts[-1]
        member = getattr(submod, leaf, None)
        if member is None:
            return None
        if inspect.isroutine(member):
            return ResolvedFn(fn=member)
        elif inspect.isclass(member):
            if not _class_allowed_by_namespace(member, spec):
                return None
            return ResolvedClass(
                cls=member, constructable=_is_constructable(member, spec)
            )
        else:
            return ResolvedObj(value=member)

    if not (include_pred(member_name) and not exclude_pred(member_name)):
        return None
    member = getattr(mod, member_name, None)
    if member is None:
        return None
    # If the resolved member is actually a submodule (ModuleType), return it as
    # a ResolvedObj. The caller (attribute access vs import-from) can impose
    # additional policy (e.g., recursion required) if needed.
    if isinstance(member, ModuleType):
        return ResolvedObj(value=member)
    if inspect.isroutine(member):
        return ResolvedFn(fn=member)
    elif inspect.isclass(member):
        if not _class_allowed_by_namespace(member, spec):
            return None
        return ResolvedClass(cls=member, constructable=_is_constructable(member, spec))
    else:
        return ResolvedObj(value=member)


def _resolve_inherited_member(
    spec: Namespace,
    member_name: str,
    include_pred,
    exclude_pred,
    scope: ResolutionScope,
) -> ResolvedFn | ResolvedClass | ResolvedObj | None:
    if not spec.parent:
        return None
    parent_res = resolve_member(spec.parent, member_name, scope)
    if parent_res is None:
        return None
    if isinstance(parent_res, ResolvedClass):
        cls_name = parent_res.cls.__name__
        if not (include_pred(cls_name) and not exclude_pred(cls_name)):
            return None
        return parent_res
    else:
        leaf = member_name.split(".")[-1]
        if include_pred(leaf) and not exclude_pred(leaf):
            return parent_res
        return None


NS_HANDLERS = {
    "virtual": _resolve_virtual_member,
    "instance": _resolve_instance_member,
    "inherited": _resolve_inherited_member,
    "module": _resolve_module_member,
}


def resolve_member(
    spec: Namespace, member_name: str, scope: ResolutionScope
) -> ResolvedFn | ResolvedClass | ResolvedObj | None:
    if handler := NS_HANDLERS.get(spec.kind):
        include_pred = make_predicate(spec.include)
        exclude_pred = make_predicate(spec.exclude)
        return handler(spec, member_name, include_pred, exclude_pred, scope)
    return None


def resolve_class(
    py_cls: type, member_name: str | None, scope: ResolutionScope
) -> ResolvedClass | ResolvedFn | ResolvedObj | None:
    main_ns = scope.namespaces.get("__main__")
    if main_ns and main_ns.kind == "virtual":
        for rc in main_ns.classes.values():
            if rc.cls is py_cls:
                return (
                    rc
                    if member_name is None
                    else _resolve_class_member(py_cls, member_name, main_ns)
                )

    class_mod = getattr(py_cls, "__module__", "")
    candidates: list[tuple[int, Namespace]] = []
    # Use module_index for direct/prefix matches first
    lists: list[Namespace] = []
    if class_mod in scope.module_index:
        lists.extend(scope.module_index[class_mod])
    else:
        parts = class_mod.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in scope.module_index:
                lists.extend(scope.module_index[prefix])
                break
    for ns in lists:
        mpath = (
            ns.module
            if isinstance(ns.module, str)
            else (ns.module.__name__ if isinstance(ns.module, ModuleType) else None)
        )
        if not mpath:
            continue
        if class_mod == mpath or class_mod.startswith(mpath + "."):
            candidates.append((len(mpath), ns))

    for _, spec in sorted(candidates, key=lambda t: t[0], reverse=True):
        if not _class_allowed_by_namespace(py_cls, spec):
            continue
        if member_name is None:
            return _build_registered_class(py_cls, spec)
        member_res = _resolve_class_member(py_cls, member_name, spec)
        if member_res:
            return member_res
    return None
