import inspect
from typing import Any

from tic.agent import Agent, ExitClarify, ExitFail, ExitSuccess
from tic.eval.base import BaseEvaluator
from tic.eval.functions import NativeFunction
from tic.eval.objects import PrintTuple, TicDataClass, TicModule, TicObject
from tic.eval.user_errors import (
    TicError,
    TicIndexError,
    TicKeyError,
    TicTypeError,
    TicValueError,
)
from tic.eval.utils import find_class_spec, get_allowed_attributes_for_instance

MAX_RANGE_SIZE = 10_000


# A simple placeholder object to act as the @dataclass decorator.
# Its only purpose is to be recognized by the evaluator.
class _DataclassDecorator:
    pass


dataclass = _DataclassDecorator()


class _TicTypePlaceholder:
    """
    A callable, safe placeholder for native Python types to prevent sandbox escapes.
    Instead of giving the user access to the raw `type` object, we give them
    this safe placeholder. It can be called like a constructor, but it doesn't
    expose dangerous attributes like `__subclasses__`.
    """

    def __init__(self, wrapped_type: type):
        self._wrapped_type = wrapped_type
        # To make it look like a type, we'll copy its name.
        self.__name__ = wrapped_type.__name__

    def __call__(self, *args, **kwargs):
        # Delegate the call to the real type constructor.
        return self._wrapped_type(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<class '{self.__name__}'>"


def _tic_isinstance(obj: Any, class_or_tuple: Any) -> bool:
    """Custom isinstance function for the tic evaluator."""
    if isinstance(class_or_tuple, _TicTypePlaceholder):
        return isinstance(obj, class_or_tuple._wrapped_type)
    if isinstance(class_or_tuple, TicDataClass):
        if isinstance(obj, TicObject):
            return obj.cls is class_or_tuple
        return False
    if isinstance(class_or_tuple, type):
        return isinstance(obj, class_or_tuple)

    # TODO: Handle tuple of types
    raise TicTypeError("isinstance() arg 2 must be a type or a tuple of types")


def _tic_type(obj: Any) -> _TicTypePlaceholder:
    """
    Sandboxed version of the `type()` built-in.

    To prevent sandbox escapes, this function returns a `_TicTypePlaceholder`
    containing the *name* of the type, rather than the type object itself.
    """
    return _TicTypePlaceholder(type(obj))


def _constrained_range(*args, **kwargs):
    """A wrapper around range() that enforces a maximum size."""
    if kwargs:
        raise TypeError("range() does not take keyword arguments.")
    r = range(*args)
    if len(r) > MAX_RANGE_SIZE:
        raise ValueError(f"Range exceeds maximum size of {MAX_RANGE_SIZE}")
    return list(r)


def _dir(evaluator: BaseEvaluator, *args, **kwargs) -> None:
    """Implementation of the dir() builtin."""
    if kwargs:
        raise TicError("dir() does not take keyword arguments.")
    if len(args) > 1:
        raise TicError(f"dir() takes at most 1 argument ({len(args)} given)")

    obj = args[0] if args else None

    attrs: tuple[str, ...]
    if obj is None:
        # If no object, dir() lists names in the current scope.
        attrs = tuple(sorted(evaluator.state.keys()))
    elif isinstance(obj, TicObject):
        attrs = tuple(sorted(obj.attributes.keys()))
    elif isinstance(obj, TicModule):
        # For modules, list their sandboxed contents.
        attrs = tuple(sorted([a for a in dir(obj) if not a.startswith("_")]))
    else:
        # Check registered classes and native types
        allowed = get_allowed_attributes_for_instance(evaluator.agent, obj)
        attrs = tuple(sorted(list(allowed)))

    current_stdout = evaluator.state.get("__stdout__")
    if not isinstance(current_stdout, list):
        current_stdout = []

    # The result of dir() is a list of strings. We wrap it in a PrintTuple
    # to match the behavior of how print() writes to stdout.
    new_stdout = current_stdout + [PrintTuple((list(attrs),))]
    evaluator.state.set("__stdout__", new_stdout)


def _hasattr(evaluator: BaseEvaluator, *args, **kwargs) -> bool:
    """Implementation of the hasattr() builtin."""
    if kwargs:
        raise TicError("hasattr() does not take keyword arguments.")
    if len(args) != 2:
        raise TicError(f"hasattr() takes exactly 2 arguments ({len(args)} given)")

    obj, name = args
    if not isinstance(name, str):
        raise TicError("hasattr(): attribute name must be a string")

    # Handle TicObjects first
    if isinstance(obj, TicObject):
        return name in obj.attributes
    if isinstance(obj, TicModule):
        return hasattr(obj, name) and not name.startswith("_")

    # Check registered classes and native types
    allowed = get_allowed_attributes_for_instance(evaluator.agent, obj)
    return name in allowed


def _get_general_help_text(agent: "Agent") -> str:
    """Returns a string with a summary of all registered items."""
    parts = []

    # Functions
    fns = sorted([name for name in agent.fn_registry.keys()])
    if fns:
        parts.append("Functions:")
        for name in fns:
            parts.append(f"  - {name}")

    # Classes
    if agent.cls_registry:
        if parts:
            parts.append("")  # Add a blank line for separation
        parts.append("Classes:")
        for name in sorted(agent.cls_registry.keys()):
            parts.append(f"  - {name}")

    # Modules
    if agent.importable_modules:
        if parts:
            parts.append("")  # Add a blank line for separation
        parts.append("Modules:")
        for name in sorted(agent.importable_modules.keys()):
            parts.append(f"  - {name}")

    if not parts:
        return "No functions, classes, or modules are registered with the agent."

    return "Available items:\n" + "\n".join(parts)


def _help(evaluator: BaseEvaluator, *args, **kwargs) -> None:
    """Implementation of the help() builtin."""
    if kwargs:
        raise TicError("help() does not take keyword arguments.")
    if len(args) > 1:
        raise TicError(f"help() takes at most 1 argument ({len(args)} given)")

    obj = args[0] if args else None

    doc = None
    if obj is None:
        doc = _get_general_help_text(evaluator.agent)
    elif isinstance(obj, TicModule):
        # Special handling to render help for a TicModule
        parts = ["Help on module " + obj.__name__ + ":\n"]
        # Introspect the module for contents
        contents = sorted([attr for attr in dir(obj) if not attr.startswith("_")])
        if contents:
            parts.append("CONTENTS")
            parts.extend([f"    {item}" for item in contents])
        doc = "\n".join(parts)
    elif isinstance(obj, (TicObject, NativeFunction)):
        doc = inspect.getdoc(obj)
    else:
        # It's a raw python object, see if we can find a docstring.
        spec = find_class_spec(evaluator.agent, obj)
        if spec:
            doc = spec.cls.__doc__

    if doc is None:
        # Fallback for everything else
        doc = inspect.getdoc(obj)

    if doc is None:
        doc = f"No documentation available for {obj!r}"

    current_stdout = evaluator.state.get("__stdout__")
    if not isinstance(current_stdout, list):
        current_stdout = []
    new_stdout = current_stdout + [PrintTuple((doc,))]
    evaluator.state.set("__stdout__", new_stdout)


BUILTINS: dict[str, Any] = {
    "print": lambda *args: PrintTuple(args),
    "len": len,
    "max": max,
    "min": min,
    "sum": sum,
    "str": _TicTypePlaceholder(str),
    "int": _TicTypePlaceholder(int),
    "float": _TicTypePlaceholder(float),
    "bool": _TicTypePlaceholder(bool),
    "dict": _TicTypePlaceholder(dict),
    "set": _TicTypePlaceholder(set),
    "tuple": _TicTypePlaceholder(tuple),
    "list": _TicTypePlaceholder(list),
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
    "type": _tic_type,
    # Dataclasses
    "dataclass": dataclass,
    # User-level exceptions, mapped from Python's names
    "Exception": TicError,
    "ValueError": TicValueError,
    "TypeError": TicTypeError,
    "KeyError": TicKeyError,
    "IndexError": TicIndexError,
    # Agent exit signals
    "exit": ExitSuccess,
    "exit_success": ExitSuccess,
    "exit_fail": ExitFail,
    "exit_clarify": ExitClarify,
    # Sandbox-aware builtins
    "dir": _dir,
    "hasattr": _hasattr,
    "help": _help,
}
