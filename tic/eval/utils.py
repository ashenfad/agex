from typing import Any, Optional

from tic.agent import Agent, RegisteredClass
from tic.eval.constants import WHITELISTED_METHODS


def find_class_spec(agent: Agent, obj: Any) -> Optional[RegisteredClass]:
    """Find the class spec for a given object instance."""
    # This helper should find the most specific spec in the MRO.
    for base in type(obj).__mro__:
        if spec := agent.cls_registry_by_type.get(base):
            return spec
    return None


def get_allowed_attributes_for_instance(agent: Agent, obj: Any) -> set[str]:
    """
    Get all allowed attributes for an object, considering its inheritance
    hierarchy and whitelisted native methods.
    """
    allowed = set()
    # Walk the MRO (Method Resolution Order) of the object's class
    for base in type(obj).__mro__:
        # Add attributes from registered classes
        if spec := agent.cls_registry_by_type.get(base):
            allowed.update(spec.attrs)
            allowed.update(spec.methods)
        # Add attributes from the native type whitelist
        if base in WHITELISTED_METHODS:
            allowed.update(WHITELISTED_METHODS[base])
    return allowed
