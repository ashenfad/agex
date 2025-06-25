import inspect
from dataclasses import dataclass
from typing import Any, Callable

from tic.agent import Agent, ExitClarify, ExitFail, ExitSuccess
from tic.eval.base import BaseEvaluator
from tic.eval.functions import UserFunction
from tic.eval.objects import (
    PrintTuple,
    TicClass,
    TicDataClass,
    TicInstance,
    TicModule,
    TicObject,
)
from tic.eval.user_errors import (
    TicArithmeticError,
    TicError,
    TicIndexError,
    TicKeyError,
    TicTypeError,
    TicValueError,
    TicZeroDivisionError,
)
from tic.eval.utils import get_allowed_attributes_for_instance
from tic.state import State


# A simple placeholder object to act as the @dataclass decorator.
# Its only purpose is to be recognized by the evaluator.
class _DataclassDecorator:
    pass


@dataclass
class StatefulFn:
    """A wrapper for stateful builtins to declare their dependencies."""

    fn: Callable[..., Any]
    needs_evaluator: bool = False


def _print_stateful(*args: Any, state: State):
    """
    A custom implementation of 'print' that appends its arguments to the
    `__stdout__` list in the agent's state as a single `PrintTuple`.
    """
    # Ensure __stdout__ exists and is a list
    current_stdout = state.get("__stdout__")
    if not isinstance(current_stdout, list):
        current_stdout = []

    # Append all arguments as a single entry
    new_stdout = current_stdout + [PrintTuple(args)]
    state.set("__stdout__", new_stdout)


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
    if isinstance(class_or_tuple, TicClass):
        # Handle user-defined classes
        if isinstance(obj, TicInstance):
            return obj.cls is class_or_tuple
        return False
    if isinstance(class_or_tuple, type):
        return isinstance(obj, class_or_tuple)

    # Handle tuple of types
    if isinstance(class_or_tuple, (tuple, list)):
        # Check each type in the tuple/list
        for single_type in class_or_tuple:
            if _tic_isinstance(obj, single_type):
                return True
        return False

    raise TicTypeError("isinstance() arg 2 must be a type or a tuple of types")


def _tic_type(obj: Any) -> _TicTypePlaceholder:
    """
    Sandboxed version of the `type()` built-in.

    To prevent sandbox escapes, this function returns a `_TicTypePlaceholder`
    containing the *name* of the type, rather than the type object itself.
    """
    return _TicTypePlaceholder(type(obj))


def _dir(evaluator: BaseEvaluator, *args, **kwargs) -> list[str]:
    """
    Implementation of the dir() builtin.
    NOTE: This is not like Python's dir(). It always prints to stdout and
    returns the list of attributes.
    """
    if kwargs:
        raise TicError("dir() does not take keyword arguments.")
    if len(args) > 1:
        raise TicError(f"dir() takes at most 1 argument ({len(args)} given)")

    obj = args[0] if args else None

    attrs: list[str]
    if obj is None:
        # If no object, dir() lists names in the current scope.
        attrs = sorted(evaluator.state.keys())
    elif isinstance(obj, TicInstance):
        # Instance attributes and class methods
        instance_attrs = set(obj.attributes.keys())
        class_methods = set(obj.cls.methods.keys())
        attrs = sorted(list(instance_attrs.union(class_methods)))
    elif isinstance(obj, TicClass):
        # Class methods
        attrs = sorted(obj.methods.keys())
    elif isinstance(obj, TicObject):
        attrs = sorted(obj.attributes.keys())
    elif isinstance(obj, TicModule):
        # For modules, introspect the actual registered module
        reg_module = evaluator.agent.importable_modules.get(obj.name)
        if reg_module:
            attrs = sorted([a for a in dir(reg_module.module) if not a.startswith("_")])
        else:
            attrs = []
    else:
        # For all other objects, respect the agent's sandbox rules.
        allowed = get_allowed_attributes_for_instance(evaluator.agent, obj)
        attrs = sorted(list(allowed))

    current_stdout = evaluator.state.get("__stdout__")
    if not isinstance(current_stdout, list):
        current_stdout = []

    new_stdout = current_stdout + [PrintTuple((attrs,))]
    evaluator.state.set("__stdout__", new_stdout)

    return attrs


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
    if isinstance(obj, TicInstance):
        return name in obj.attributes or name in obj.cls.methods
    if isinstance(obj, TicClass):
        return name in obj.methods
    if isinstance(obj, TicModule):
        # Use JIT resolution to check if the attribute exists on the real module
        reg_module = evaluator.agent.importable_modules.get(obj.name)
        if not reg_module:
            return False
        return hasattr(reg_module.module, name) and not name.startswith("_")

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


def _format_user_function_sig(fn: UserFunction) -> str:
    """Creates a string signature for a UserFunction."""
    # This is a simplified formatter. A real one would handle more arg types.
    arg_names = [arg.arg for arg in fn.args.args]
    return f"{fn.name}({', '.join(arg_names)})"


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
    elif isinstance(obj, TicInstance):
        # For an instance, show help for its class.
        return _help(evaluator, obj.cls)
    elif isinstance(obj, TicClass):
        parts = [f"Help on class {obj.name}:\n"]
        if "__init__" in obj.methods:
            init_sig = _format_user_function_sig(obj.methods["__init__"])
            parts.append(f"{obj.name}{init_sig.replace('__init__', '', 1)}")
        else:
            parts.append(f"{obj.name}()")

        methods = sorted(obj.methods.keys())
        if methods:
            parts.append("\nMethods defined here:")
            for method_name in methods:
                method_sig = _format_user_function_sig(obj.methods[method_name])
                parts.append(f"  {method_sig}")
        doc = "\n".join(parts)
    elif isinstance(obj, TicModule):
        # Special handling to render help for a TicModule
        parts = ["Help on module " + obj.name + ":\n"]
        # Introspect the actual registered module for contents
        reg_module = evaluator.agent.importable_modules.get(obj.name)
        if reg_module:
            contents = sorted(
                [attr for attr in dir(reg_module.module) if not attr.startswith("_")]
            )
            if contents:
                parts.append("CONTENTS")
                parts.extend([f"    {item}" for item in contents])
        doc = "\n".join(parts)
    else:
        # For everything else (TicObject, NativeFunction, raw Python objects/functions),
        # just try to get a docstring.
        doc = inspect.getdoc(obj)

    if doc is None:
        doc = "No help available."

    # All help output goes to stdout
    current_stdout = evaluator.state.get("__stdout__")
    if not isinstance(current_stdout, list):
        current_stdout = []

    new_stdout = current_stdout + [PrintTuple((doc,))]
    evaluator.state.set("__stdout__", new_stdout)


STATEFUL_BUILTINS: dict[str, StatefulFn] = {
    "print": StatefulFn(_print_stateful),
    "help": StatefulFn(_help, needs_evaluator=True),
    "dir": StatefulFn(_dir, needs_evaluator=True),
    "hasattr": StatefulFn(_hasattr, needs_evaluator=True),
}


# This is the main registry of built-in functions available in the sandbox.
BUILTINS = {
    "abs": abs,
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
    "round": round,
    "all": all,
    "any": any,
    "sorted": sorted,
    "range": range,
    "reversed": reversed,
    "zip": zip,
    "enumerate": enumerate,
    "map": map,
    "filter": filter,
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
    "ZeroDivisionError": TicZeroDivisionError,
    "ArithmeticError": TicArithmeticError,
    # Agent exit signals
    "exit": ExitSuccess,
    "exit_success": ExitSuccess,
    "exit_fail": ExitFail,
    "exit_clarify": ExitClarify,
}
