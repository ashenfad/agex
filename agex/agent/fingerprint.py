import hashlib
import json
from typing import Dict

from .datatypes import RegisteredClass, RegisteredFn, RegisteredModule


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


def compute_agent_fingerprint(
    primer: str | None,
    fn_registry: Dict[str, RegisteredFn],
    cls_registry: Dict[str, RegisteredClass],
    importable_modules: Dict[str, RegisteredModule],
) -> str:
    """
    Compute a deterministic fingerprint for an agent based on its registrations.

    The fingerprint includes:
    - Function registry (names, visibility, docstrings)
    - Class registry (names, visibility, constructable, all attrs/methods)
    - Module registry (names, visibility, all fns/consts/classes)
    - Primer text
    """
    # Extract registry data in a deterministic format
    registry_data = {
        "primer": primer or "",
        "functions": {
            name: {
                "visibility": spec.visibility,
                "docstring": spec.docstring or "",
            }
            for name, spec in sorted(fn_registry.items())
        },
        "classes": {
            name: _serialize_registered_class(spec)
            for name, spec in sorted(cls_registry.items())
        },
        "modules": {
            name: _serialize_registered_module(spec)
            for name, spec in sorted(importable_modules.items())
        },
    }

    # Create deterministic JSON and hash it
    json_str = json.dumps(registry_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
