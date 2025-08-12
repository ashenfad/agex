import hashlib
import json
from typing import Any, Dict

from .datatypes import MemberSpec, RegisteredClass, RegisteredModule


def _serialize_member_spec(spec):
    """Serialize a MemberSpec into a deterministic dict."""
    return {
        "visibility": spec.visibility,
        "docstring": spec.docstring or "",
        "constructable": spec.constructable,
    }


def _serialize_registered_class(cls_spec: RegisteredClass) -> Dict:
    """Serialize a RegisteredClass with all its member specifications."""
    return {
        "visibility": cls_spec.visibility,
        "constructable": cls_spec.constructable,
        "attrs": {
            name: _serialize_member_spec(spec)
            for name, spec in sorted(cls_spec.attrs.items())
        },
        "methods": {
            name: _serialize_member_spec(spec)
            for name, spec in sorted(cls_spec.methods.items())
        },
    }


def _serialize_registered_module(mod_spec: RegisteredModule) -> Dict:
    """Serialize a RegisteredModule with all its member specifications."""
    return {
        "visibility": mod_spec.visibility,
        "name": mod_spec.name,
        "fns": {
            name: _serialize_member_spec(spec)
            for name, spec in sorted(mod_spec.fns.items())
        },
        "consts": {
            name: _serialize_member_spec(spec)
            for name, spec in sorted(mod_spec.consts.items())
        },
        "classes": {
            name: _serialize_registered_class(spec)
            for name, spec in sorted(mod_spec.classes.items())
        },
    }


def _serialize_pattern(pattern: Any) -> Any:
    """Serialize include/exclude patterns deterministically."""
    if pattern is None:
        return None
    if isinstance(pattern, str):
        return pattern
    if isinstance(pattern, (list, tuple, set)):
        # Only keep string-like values; ignore callables for determinism
        items = [str(x) for x in pattern]
        return sorted(items)
    # Callable or unknown types
    return str(pattern)


def _serialize_policy_memberspec(ms: MemberSpec) -> Dict[str, Any]:
    return {
        "visibility": ms.visibility,
        "docstring": ms.docstring or "",
        "constructable": ms.constructable,
    }


def _serialize_policy_namespace(ns) -> Dict[str, Any]:
    from types import ModuleType

    from .policy.datatypes import Namespace as PolicyNamespace  # type: ignore

    assert isinstance(ns, PolicyNamespace)
    data: Dict[str, Any] = {
        "name": ns.name,
        "kind": ns.kind,
        "visibility": ns.visibility,
        "include": _serialize_pattern(ns.include),
        "exclude": _serialize_pattern(ns.exclude),
        "recursive": bool(getattr(ns, "recursive", False)),
    }
    # Module path (string) for module kinds
    if ns.kind == "module":
        if isinstance(ns.module, str):
            data["module_path"] = ns.module
        elif isinstance(ns.module, ModuleType):
            data["module_path"] = ns.module.__name__
        else:
            data["module_path"] = None
    # Parent linkage for inherited kinds
    if ns.kind == "inherited":
        data["parent"] = getattr(ns.parent, "name", None)
    # Virtual (__main__) functions and classes
    if ns.kind == "virtual":
        data["functions"] = {
            name: _serialize_policy_memberspec(ms)
            for name, ms in sorted(ns.fns.items())
        }
        # Classes: capture minimal spec from per-class namespace if available
        classes: Dict[str, Any] = {}

        # Try to access the owning policy via back-reference in serializer caller
        # The caller will pass classes explicitly via policy._class_namespaces
        classes["__placeholder__"] = None  # will be removed by caller
        data["classes"] = classes
    return data


def compute_agent_fingerprint_from_policy(agent: Any) -> str:
    """
    Compute fingerprint from the agent's policy namespaces (capability specs).

    Includes:
    - Primer text
    - For each namespace: name, kind, module path, visibility, include/exclude, configure, recursive
    - For virtual (__main__): functions (name -> vis/doc), classes with per-class namespace specs
    - Instance namespaces exclude live object identity
    """
    policy = agent._policy  # type: ignore[attr-defined]

    # Serialize namespaces
    ns_items: Dict[str, Any] = {}
    # We'll need per-class namespaces to capture class include/exclude/configure
    per_class_ns = getattr(policy, "_class_namespaces", {})

    for name in sorted(policy.namespaces.keys()):
        ns = policy.namespaces[name]
        data = _serialize_policy_namespace(ns)
        # Remove placeholder and fill actual classes block if virtual
        if ns.kind == "virtual":
            classes: Dict[str, Any] = {}
            for cls_name, rc in sorted(ns.classes.items()):
                # Use per-class namespace if present, else fallback to defaults
                cls_ns = per_class_ns.get(rc.cls)
                cls_entry: Dict[str, Any] = {
                    "name": cls_name,
                    "module_path": getattr(rc.cls, "__module__", None),
                    "constructable": bool(getattr(rc, "constructable", True)),
                }
                if cls_ns is not None:
                    cls_entry.update(
                        {
                            "visibility": cls_ns.visibility,
                            "include": _serialize_pattern(cls_ns.include),
                            "exclude": _serialize_pattern(cls_ns.exclude),
                            "configure": {
                                k: _serialize_policy_memberspec(v)
                                for k, v in sorted(cls_ns.configure.items())
                            },
                        }
                    )
                classes[cls_name] = cls_entry
            data["classes"] = classes
        # Add generic configure for non-virtual namespaces
        data["configure"] = {
            k: _serialize_policy_memberspec(v) for k, v in sorted(ns.configure.items())
        }
        ns_items[name] = data

    payload = {
        "primer": agent.primer or "",
        "policy_namespaces": ns_items,
    }
    json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
