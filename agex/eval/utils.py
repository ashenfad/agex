import dataclasses
from typing import Any

from agex.agent.base import BaseAgent
from agex.agent.utils import get_instance_attributes_from_init
from agex.eval.constants import WHITELISTED_METHODS


def get_allowed_attributes_for_instance(agent: BaseAgent, obj: Any) -> set[str]:
    """
    Get all allowed attributes for an object, considering its inheritance
    hierarchy and whitelisted native methods.
    """
    allowed: set[str] = set()

    # Walk the MRO (Method Resolution Order) of the object's class
    for base in type(obj).__mro__:
        # Build candidate names to probe through policy
        candidate_names: set[str] = set()
        try:
            candidate_names.update(dir(base))
        except Exception:
            pass
        # Add annotated attributes (often dataclass field names)
        candidate_names.update(getattr(base, "__annotations__", {}).keys())
        # Add dataclass field names if any
        try:
            candidate_names.update({f.name for f in dataclasses.fields(base)})
        except Exception:
            pass
        # Add instance attributes discovered from __init__ assignment
        try:
            candidate_names.update(get_instance_attributes_from_init(base))
        except Exception:
            pass
        # If class was registered with explicit include list, include those names too
        try:
            cls_ns = getattr(agent._policy, "_class_namespaces", {}).get(base)  # type: ignore[attr-defined]
            if cls_ns is not None and isinstance(cls_ns.include, (list, set, tuple)):
                for item in cls_ns.include:
                    if isinstance(item, str):
                        # Handle dotted class.member includes by taking the member leaf
                        candidate_names.add(item.split(".")[-1])
        except Exception:
            pass

        # Probe visibility via policy resolver; this includes instance-only attrs now
        for name in candidate_names:
            if name.startswith("__") and name != "__init__":
                continue
            try:
                res = agent._policy.resolve_class_member(base, name)  # type: ignore[attr-defined]
            except Exception:
                res = None
            if res is not None:
                allowed.add(name)

        # Add attributes from the native type whitelist
        if base in WHITELISTED_METHODS:
            allowed.update(WHITELISTED_METHODS[base])

    return allowed
