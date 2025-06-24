from typing import Any

from ..agent import ExitClarify, ExitFail, ExitSuccess
from .objects import TicDataClass, TicObject
from .user_errors import (
    TicError,
    TicIndexError,
    TicKeyError,
    TicTypeError,
    TicValueError,
)

MAX_RANGE_SIZE = 10_000


# A simple placeholder object to act as the @dataclass decorator.
# Its only purpose is to be recognized by the evaluator.
class _DataclassDecorator:
    pass


dataclass = _DataclassDecorator()


def _tic_isinstance(obj: Any, class_or_tuple: Any) -> bool:
    """Custom isinstance function for the tic evaluator."""
    if isinstance(class_or_tuple, TicDataClass):
        if isinstance(obj, TicObject):
            return obj.cls is class_or_tuple
        return False
    # TODO: Handle tuple of types
    return isinstance(obj, class_or_tuple)


def _constrained_range(*args, **kwargs):
    """A wrapper around range() that enforces a maximum size."""
    if kwargs:
        raise TypeError("range() does not take keyword arguments.")
    r = range(*args)
    if len(r) > MAX_RANGE_SIZE:
        raise ValueError(f"Range exceeds maximum size of {MAX_RANGE_SIZE}")
    return list(r)


BUILTINS: dict[str, Any] = {
    "len": len,
    "max": max,
    "min": min,
    "sum": sum,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "list": list,
    "abs": abs,
    "round": round,
    "all": all,
    "any": any,
    "sorted": sorted,
    "range": _constrained_range,
    "reversed": lambda x: list(reversed(x)),
    "zip": lambda *args: list(zip(*args)),
    "enumerate": lambda x: list(enumerate(x)),
    "map": lambda f, it: list(map(f, it)),
    "filter": lambda f, it: list(filter(f, it)),
    # Type introspection
    "isinstance": _tic_isinstance,
    "type": type,
    # Dataclasses
    "dataclass": dataclass,
    # User-level exceptions, mapped from Python's names
    "Exception": TicError,
    "ValueError": TicValueError,
    "TypeError": TicTypeError,
    "KeyError": TicKeyError,
    "IndexError": TicIndexError,
    # Agent exit signals
    "exit_success": ExitSuccess,
    "exit_fail": ExitFail,
    "exit_clarify": ExitClarify,
}
