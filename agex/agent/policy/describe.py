from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

from agex.agent.utils import get_instance_attributes_from_init

from ..datatypes import MemberSpec
from .datatypes import Namespace, Visibility
from .resolve import _is_constructable


@dataclass
class Description:
    kind: Literal["fn", "class", "obj", "module"]
    visibility: Visibility | None
    docstring: str | None
    constructable: bool | None = None
    members: dict[str, "Description"] | None = None


def get_effective_member_spec(
    ns: Namespace, *, class_name: str | None, member_name: str
) -> MemberSpec:
    """
    Returns the configured MemberSpec for a given member, applying dotted-key
    precedence (e.g., "Class.member" overrides "member"). Falls back to an
    empty MemberSpec when not configured.
    """
    if class_name:
        dotted_key = f"{class_name}.{member_name}"
        return ns.configure.get(dotted_key, ns.configure.get(member_name, MemberSpec()))
    return ns.configure.get(member_name, MemberSpec())


def collect_class_candidate_names(
    py_cls: type, *, ns: Namespace, constructable: bool
) -> set[str]:
    """
    Build the set of candidate member names for a class using the policy's
    include/exclude intent. Considers:
    - public attributes and methods on the class (allowing __init__)
    - class-level annotations
    - instance attributes assigned in __init__ when wildcards or explicit
      includes are present
    - explicit include lists
    It also enforces constructability on __init__.
    """
    candidate_names: set[str] = set()
    for name, _obj in inspect.getmembers(py_cls):
        if name.startswith("__") and name != "__init__":
            continue
        candidate_names.add(name)
    if hasattr(py_cls, "__annotations__"):
        candidate_names.update(getattr(py_cls, "__annotations__").keys())
    # Include instance attributes assigned in __init__ when the include pattern
    # clearly applies to this class:
    # - Global wildcards when not describing under a concrete module namespace
    # - Or a dotted wildcard like "ClassName.*" for this specific class
    include_instance_attrs = False
    if isinstance(ns.include, str):
        if ns.include == "*" or "*" in ns.include:
            # Only treat undotted wildcard as class-applicable when not under a module
            include_instance_attrs = ns.kind != "module"
    elif isinstance(ns.include, (list, set, tuple)):
        for it in ns.include:
            if (
                isinstance(it, str)
                and it.startswith(f"{py_cls.__name__}.")
                and "*" in it
            ):
                include_instance_attrs = True
                break
    if include_instance_attrs:
        try:
            candidate_names.update(get_instance_attributes_from_init(py_cls))
        except Exception:
            pass
    # Add explicit includes that are relevant to this class:
    # - Dotted entries like "Class.member" matching this class
    # - Undotted entries only when not describing under a module namespace
    if isinstance(ns.include, (list, set, tuple)):
        for it in ns.include:
            if not isinstance(it, str):
                continue
            if "." in it:
                cls, _, short = it.partition(".")
                if cls == py_cls.__name__ and short:
                    # Ignore wildcard fragments like "*" or patterns
                    if any(ch in short for ch in "*?[]"):
                        continue
                    candidate_names.add(short)
            else:
                if ns.kind != "module":
                    candidate_names.add(it)
    # Enforce constructability on __init__
    if constructable:
        candidate_names.add("__init__")
    else:
        candidate_names.discard("__init__")
    return candidate_names


def _effective_visibility(ns: Namespace, key: str) -> Visibility | None:
    ms = ns.configure.get(key, MemberSpec())
    return ms.visibility or ns.visibility


def describe_member(ns: Namespace, member_path: str) -> Description | None:
    # Inherited namespaces: ensure child allows, then delegate to parent
    if ns.kind == "inherited" and ns.parent is not None:
        leaf = member_path.split(".")[-1]
        # Apply child's include/exclude via simple predicate
        from .resolve import make_predicate  # local import to avoid cycles

        include_pred = make_predicate(ns.include)
        exclude_pred = make_predicate(ns.exclude)
        if not (include_pred(leaf) and not exclude_pred(leaf)):
            return None
        return describe_member(ns.parent, member_path)

    # Class member
    if "." in member_path:
        if ns.kind == "module":
            mod = ns._ensure_module_loaded()
            cls_name, short = member_path.split(".", 1)
            cls_obj = getattr(mod, cls_name, None)
            if not inspect.isclass(cls_obj):
                return None
            obj = getattr(cls_obj, short, None)
            if obj is None:
                return None
            kind = "fn" if inspect.isroutine(obj) else "obj"
            vis = _effective_visibility(ns, f"{cls_name}.{short}")
            doc = ns.configure.get(
                f"{cls_name}.{short}", MemberSpec()
            ).docstring or getattr(obj, "__doc__", None)
            return Description(
                kind=kind, visibility=vis, docstring=doc, constructable=None
            )
        elif ns.kind == "virtual":
            cls_name, short = member_path.split(".", 1)
            rc = ns.classes.get(cls_name)
            if not rc:
                return None
            obj = getattr(rc.cls, short, None)
            if obj is None:
                return None
            kind = "fn" if inspect.isroutine(obj) else "obj"
            vis = _effective_visibility(ns, f"{cls_name}.{short}")
            doc = ns.configure.get(
                f"{cls_name}.{short}", MemberSpec()
            ).docstring or getattr(obj, "__doc__", None)
            return Description(
                kind=kind, visibility=vis, docstring=doc, constructable=None
            )
        elif ns.kind == "instance":
            # Class.member not applicable to instances
            return None

    # Top-level member
    name = member_path
    if ns.kind == "module":
        mod = ns._ensure_module_loaded()
        obj = getattr(mod, name, None)
        if obj is None:
            return None
        if inspect.isclass(obj):
            vis = _effective_visibility(ns, name)
            doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                obj, "__doc__", None
            )
            constructable = _is_constructable(obj, ns)
            return Description(
                kind="class", visibility=vis, docstring=doc, constructable=constructable
            )
        if isinstance(obj, ModuleType):
            vis = _effective_visibility(ns, name)
            doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                obj, "__doc__", None
            )
            return Description(kind="module", visibility=vis, docstring=doc)
        kind = "fn" if inspect.isroutine(obj) else "obj"
        vis = _effective_visibility(ns, name)
        doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
            obj, "__doc__", None
        )
        return Description(kind=kind, visibility=vis, docstring=doc, constructable=None)
    elif ns.kind == "instance":
        if ns.obj is None:
            return None
        obj = getattr(ns.obj, name, None)
        if obj is None:
            return None
        kind = "fn" if callable(obj) else "obj"
        vis = _effective_visibility(ns, name)
        doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
            obj, "__doc__", None
        )
        return Description(kind=kind, visibility=vis, docstring=doc, constructable=None)
    elif ns.kind == "virtual":
        # Functions
        if name in ns.fns:
            fn = ns.fn_objects.get(name)
            vis = _effective_visibility(ns, name)
            doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                fn, "__doc__", None
            )
            return Description(
                kind="fn", visibility=vis, docstring=doc, constructable=None
            )
        # Classes
        if name in ns.classes:
            rc = ns.classes[name]
            vis = _effective_visibility(ns, name)
            doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                rc.cls, "__doc__", None
            )
            return Description(
                kind="class",
                visibility=vis,
                docstring=doc,
                constructable=rc.constructable,
            )
        return None
    else:
        # inherited handled above
        return None


def describe_class(
    py_cls: type, ns: Namespace, include_low: bool = False
) -> Description:
    vis = _effective_visibility(ns, py_cls.__name__)
    doc = ns.configure.get(py_cls.__name__, MemberSpec()).docstring or getattr(
        py_cls, "__doc__", None
    )
    constructable = _is_constructable(py_cls, ns)
    # Enumerate visible members of class (non-recursive)
    from .resolve import make_predicate  # local import to avoid cycles

    include_pred = make_predicate(ns.include)
    exclude_pred = make_predicate(ns.exclude)

    candidate_names = collect_class_candidate_names(
        py_cls, ns=ns, constructable=constructable
    )

    members: dict[str, Description] = {}
    for name in sorted(candidate_names):
        # Apply include/exclude on plain or dotted key, but allow __init__ if constructable
        if name != "__init__":
            dotted = f"{py_cls.__name__}.{name}"
            inc_ok = include_pred(name) or include_pred(dotted)
            exc_hit = exclude_pred(name) or exclude_pred(dotted)
            if not (inc_ok and not exc_hit):
                continue
        # Effective visibility uses dotted override first, falling back to plain
        ms = get_effective_member_spec(ns, class_name=py_cls.__name__, member_name=name)
        eff_vis = ms.visibility or ns.visibility
        if eff_vis == "low" and not include_low:
            continue
        obj = getattr(py_cls, name, None)
        if name == "__init__" or inspect.isroutine(obj):
            doc_m = ms.docstring or getattr(obj, "__doc__", None)
            members[name] = Description(kind="fn", visibility=eff_vis, docstring=doc_m)
        else:
            doc_a = ms.docstring or getattr(obj, "__doc__", None)
            members[name] = Description(kind="obj", visibility=eff_vis, docstring=doc_a)

    return Description(
        kind="class",
        visibility=vis,
        docstring=doc,
        constructable=constructable,
        members=members or None,
    )


def describe_namespace(
    ns: Namespace, include_low: bool = False
) -> dict[str, Description]:
    # Inherited namespaces: compute view over parent with child's filters
    from .resolve import make_predicate  # local import to avoid cycles

    inc = make_predicate(ns.include)
    exc = make_predicate(ns.exclude)

    def include_key(plain_key: str, dotted_prefix: str | None = None) -> bool:
        """
        Apply include/exclude to either the plain key or dotted prefix.key. Visibility
        is determined by the plain key.
        """
        eff_vis = _effective_visibility(ns, plain_key)
        if eff_vis == "low" and not include_low:
            return False
        dotted_key = f"{dotted_prefix}.{plain_key}" if dotted_prefix else None
        inc_ok = inc(plain_key) or (dotted_key is not None and inc(dotted_key))
        exc_hit = exc(plain_key) or (dotted_key is not None and exc(dotted_key))
        return inc_ok and not exc_hit

    result: dict[str, Description] = {}

    if ns.kind == "inherited" and ns.parent is not None:
        parent = ns.parent
        # Walk parent's top-level view but apply child's filters/overrides
        if parent.kind == "module":
            # Intersect parent's allowed view with child's include/exclude
            parent_view = describe_namespace(parent, include_low)
            for name, desc in parent_view.items():
                dotted_prefix = None
                try:
                    dotted_prefix = parent._ensure_module_loaded().__name__
                except Exception:
                    dotted_prefix = None
                if not include_key(name, dotted_prefix):
                    continue
                # Keep parent's kind/doc; apply child's effective visibility if configured
                eff_vis = _effective_visibility(ns, name)
                if desc.kind == "class":
                    # Re-describe class members with child's ns for dotted filters
                    cls_obj = getattr(parent._ensure_module_loaded(), name, None)
                    if inspect.isclass(cls_obj):
                        result[name] = describe_class(cls_obj, ns, include_low)
                    else:
                        # Fallback to parent's entry
                        result[name] = Description(
                            kind=desc.kind,
                            visibility=eff_vis,
                            docstring=desc.docstring,
                            constructable=desc.constructable,
                        )
                else:
                    result[name] = Description(
                        kind=desc.kind,
                        visibility=eff_vis,
                        docstring=desc.docstring,
                        constructable=desc.constructable,
                    )
            return result
        elif parent.kind == "virtual":
            for name in parent.fns.keys():
                if include_key(name):
                    fn = parent.fn_objects.get(name)
                    doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                        fn, "__doc__", None
                    )
                    result[name] = Description(
                        kind="fn",
                        visibility=_effective_visibility(ns, name),
                        docstring=doc,
                    )
            for name, rc in parent.classes.items():
                if include_key(name):
                    result[name] = describe_class(rc.cls, ns, include_low)
            return result
        elif parent.kind == "instance":
            # Describe from parent's instance but filtered by child
            if parent.obj is None:
                return result
            for name, obj in inspect.getmembers(parent.obj):
                if not include_key(name):
                    continue
                kind = "fn" if callable(obj) else "obj"
                doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                    obj, "__doc__", None
                )
                result[name] = Description(
                    kind=kind, visibility=_effective_visibility(ns, name), docstring=doc
                )
            return result

    # Non-inherited namespaces
    if ns.kind == "module":
        mod = ns._ensure_module_loaded()
        mod_name = getattr(mod, "__name__", None)
        for name, obj in inspect.getmembers(mod):
            if name.startswith("@"):
                continue
            if not include_key(name, mod_name):
                continue
            if inspect.isclass(obj):
                result[name] = describe_class(obj, ns, include_low)
            elif inspect.isroutine(obj):
                doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                    obj, "__doc__", None
                )
                result[name] = Description(
                    kind="fn", visibility=_effective_visibility(ns, name), docstring=doc
                )
            elif isinstance(obj, ModuleType):
                doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                    obj, "__doc__", None
                )
                result[name] = Description(
                    kind="module",
                    visibility=_effective_visibility(ns, name),
                    docstring=doc,
                )
            else:
                doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                    obj, "__doc__", None
                )
                result[name] = Description(
                    kind="obj",
                    visibility=_effective_visibility(ns, name),
                    docstring=doc,
                )
        return result
    elif ns.kind == "virtual":
        for name, ms in ns.fns.items():
            if include_key(name):
                fn = ns.fn_objects.get(name)
                doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                    fn, "__doc__", None
                )
                result[name] = Description(
                    kind="fn", visibility=_effective_visibility(ns, name), docstring=doc
                )
        for name, rc in ns.classes.items():
            if include_key(name):
                result[name] = describe_class(rc.cls, ns, include_low)
        return result
    elif ns.kind == "instance":
        # Use instance object; avoid invoking properties if possible; docstring best-effort
        if ns.obj is None:
            return result
        for name in dir(ns.obj):
            if not include_key(name):
                continue
            try:
                obj = getattr(ns.obj, name)
            except Exception:
                continue
            kind = "fn" if callable(obj) else "obj"
            doc = ns.configure.get(name, MemberSpec()).docstring or getattr(
                obj, "__doc__", None
            )
            result[name] = Description(
                kind=kind, visibility=_effective_visibility(ns, name), docstring=doc
            )
        return result
    elif ns.kind == "inherited":
        # parent None case already handled above
        return result

    return result
