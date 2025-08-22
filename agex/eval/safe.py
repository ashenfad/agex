"""
Pickle safety checking for assignment operations.

This module provides utilities to check if values are safe to assign
before they enter the state, catching unpickleable objects early with
good error messages.
"""

import io
import pickle
from types import ModuleType
from typing import Any

from .error import EvalError
from .objects import AgexModule


def check_assignment_safety(value: Any) -> Any:
    """
    Check if a value is safe to assign, with performance optimizations.

    Uses fast heuristics to avoid expensive pickle tests when possible.
    Converts known problematic types (like modules) to safe equivalents.

    Args:
        value: The value being assigned

    Returns:
        The value (possibly converted to a safe equivalent)

    Raises:
        EvalError: If the value is not pickleable
    """
    # Convert modules immediately (known issue)
    if isinstance(value, ModuleType):
        return AgexModule(name=value.__name__, agent_fingerprint="")

    # Fast path: known-safe atomic types
    safe_atomic_types = {int, float, str, bytes, bool, type(None), complex, range}
    if type(value) in safe_atomic_types:
        return value

    # Explicitly block file objects
    if isinstance(value, io.IOBase):
        raise EvalError(
            f"Cannot assign file object of type {type(value).__name__} to state.",
            node=None,
        )

    # Fast path: objects with pickle dunders that aren't collections
    if _has_pickle_support(value) and not _is_collection(value):
        return value

    # Dataclasses are usually safe (and have __dataclass_fields__)
    if hasattr(value, "__dataclass_fields__"):
        return value

    # NumPy objects are generally safe
    if hasattr(value, "__module__") and "numpy" in str(
        getattr(value, "__module__", "")
    ):
        return value

    # Provide helpful error messages for common unpickleable types
    type_name = type(value).__name__
    if type_name in ("dict_keys", "dict_values", "dict_items"):
        method_name = type_name.replace("dict_", "")
        raise EvalError(
            f"Cannot assign {type_name} object. Use list(dict.{method_name}()) to convert to a list.",
            node=None,
        )

    if type_name == "map":
        raise EvalError(
            "Cannot assign map object. Use list(map(...)) to convert to a list.",
            node=None,
        )

    if type_name == "filter":
        raise EvalError(
            "Cannot assign filter object. Use list(filter(...)) to convert to a list.",
            node=None,
        )

    if type_name == "enumerate":
        raise EvalError(
            "Cannot assign enumerate object. Use list(enumerate(...)) to convert to a list.",
            node=None,
        )

    # Fallback to a full pickle test
    try:
        pickle.dumps(value)
    except Exception:
        raise EvalError(
            f"Cannot assign unpickleable object of type {type(value).__name__}",
            node=None,
        )

    return value


def _has_pickle_support(obj: Any) -> bool:
    """Check if object explicitly supports pickling via dunder methods."""
    obj_type = type(obj)
    # Use only the most reliable pickle methods that strongly indicate
    # the object was designed to be pickled
    reliable_pickle_methods = [
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
    ]

    for method in reliable_pickle_methods:
        # Check if method is defined on the type (not inherited from object)
        if method in obj_type.__dict__:
            return True

    return False


def _is_collection(obj: Any) -> bool:
    """Check if object is a collection that could contain other objects."""
    return isinstance(obj, (list, tuple, dict, set, frozenset))
