import hashlib
import json
from typing import Any, Dict

from .datatypes import MemberSpec


def _serialize_pattern(pattern: Any) -> Any:
    """Serialize include/exclude patterns deterministically.

    Only includes string patterns - callables are ignored since they
    can't be serialized deterministically (memory addresses change).
    """
    if pattern is None:
        return None
    if isinstance(pattern, str):
        return pattern
    if isinstance(pattern, (list, tuple, set)):
        # Only keep string values; skip callables entirely
        items = [x for x in pattern if isinstance(x, str)]
        return sorted(items) if items else None
    # Callable or unknown types - skip entirely
    return None


def _serialize_memberspec(ms: MemberSpec) -> Dict[str, Any]:
    """Serialize a MemberSpec - only user-provided configuration."""
    return {
        "visibility": ms.visibility,
        "docstring": ms.docstring or "",
        "constructable": ms.constructable,
    }


def _serialize_namespace(ns: Any) -> Dict[str, Any]:
    """Serialize a namespace's configuration (not discovered content)."""
    from types import ModuleType

    data: Dict[str, Any] = {
        "name": ns.name,
        "kind": ns.kind,
        "visibility": ns.visibility,
        "include": _serialize_pattern(ns.include),
        "exclude": _serialize_pattern(ns.exclude),
        "recursive": bool(getattr(ns, "recursive", False)),
    }

    # Module path for module namespaces
    if ns.kind == "module":
        if isinstance(ns.module, str):
            data["module_path"] = ns.module
        elif isinstance(ns.module, ModuleType):
            data["module_path"] = ns.module.__name__
        else:
            data["module_path"] = None

    # Parent linkage for inherited namespaces
    if ns.kind == "inherited":
        data["parent"] = getattr(ns.parent, "name", None)

    # User-provided configure overrides (explicit MemberSpec settings)
    if ns.configure:
        data["configure"] = {
            k: _serialize_memberspec(v) for k, v in sorted(ns.configure.items())
        }

    return data


def compute_agent_fingerprint_from_policy(agent: Any) -> str:
    """
    Compute fingerprint from the agent's explicit configuration only.

    Includes:
    - Agent name
    - Primer text
    - For each namespace: name, kind, module path, visibility,
      include/exclude patterns, configure overrides, recursive flag

    Does NOT include:
    - Discovered module members
    - Runtime state (ns.fns, ns.classes contents)
    - Docstrings discovered from objects
    - Class __module__ attributes
    """
    policy = agent._policy

    # Serialize only the configuration, not discovered content
    ns_items: Dict[str, Any] = {}
    for name in sorted(policy.namespaces.keys()):
        ns = policy.namespaces[name]
        ns_items[name] = _serialize_namespace(ns)

    payload = {
        "name": agent.name or "",
        "primer": agent.primer or "",
        "namespaces": ns_items,
    }

    json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
