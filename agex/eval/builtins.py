import inspect
from dataclasses import dataclass
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.agent.datatypes import ExitClarify, ExitFail, ExitSuccess
from agex.eval.base import BaseEvaluator
from agex.eval.functions import UserFunction
from agex.eval.objects import (
    AgexClass,
    AgexDataClass,
    AgexInstance,
    AgexModule,
    AgexObject,
    PrintTuple,
)
from agex.eval.user_errors import (
    AgexArithmeticError,
    AgexError,
    AgexIndexError,
    AgexKeyError,
    AgexTypeError,
    AgexValueError,
    AgexZeroDivisionError,
)
from agex.eval.utils import get_allowed_attributes_for_instance
from agex.state import State


def _is_bound_instance_object(obj: Any) -> bool:
    """Check if an object is a BoundInstanceObject (registered live object)."""
    return (
        hasattr(obj, "reg_object")
        and hasattr(obj.reg_object, "methods")
        and hasattr(obj.reg_object, "properties")
    )


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


class _AgexTypePlaceholder:
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


def _agex_isinstance(obj: Any, class_or_tuple: Any) -> bool:
    """Custom isinstance function for the tic evaluator."""
    if isinstance(class_or_tuple, _AgexTypePlaceholder):
        return isinstance(obj, class_or_tuple._wrapped_type)
    if isinstance(class_or_tuple, AgexDataClass):
        if isinstance(obj, AgexObject):
            return obj.cls is class_or_tuple
        return False
    if isinstance(class_or_tuple, AgexClass):
        # Handle user-defined classes
        if isinstance(obj, AgexInstance):
            return obj.cls is class_or_tuple
        return False
    if isinstance(class_or_tuple, type):
        return isinstance(obj, class_or_tuple)

    # Handle tuple of types
    if isinstance(class_or_tuple, (tuple, list)):
        # Check each type in the tuple/list
        for single_type in class_or_tuple:
            if _agex_isinstance(obj, single_type):
                return True
        return False

    raise AgexTypeError("isinstance() arg 2 must be a type or a tuple of types")


def _agex_type(obj: Any) -> _AgexTypePlaceholder:
    """
    Sandboxed version of the `type()` built-in.

    To prevent sandbox escapes, this function returns a `_AgexTypePlaceholder`
    containing the *name* of the type, rather than the type object itself.
    """
    return _AgexTypePlaceholder(type(obj))


def _dir(evaluator: BaseEvaluator, *args, **kwargs) -> list[str]:
    """
    Implementation of the dir() builtin.
    NOTE: This is not like Python's dir(). It always prints to stdout and
    returns the list of attributes.
    """
    if kwargs:
        raise AgexError("dir() does not take keyword arguments.")
    if len(args) > 1:
        raise AgexError(f"dir() takes at most 1 argument ({len(args)} given)")

    obj = args[0] if args else None

    attrs: list[str]
    if obj is None:
        # If no object, dir() lists names in the current scope.
        attrs = sorted(evaluator.state.keys())
    elif isinstance(obj, AgexInstance):
        # Instance attributes and class methods
        instance_attrs = set(obj.attributes.keys())
        class_methods = set(obj.cls.methods.keys())
        attrs = sorted(list(instance_attrs.union(class_methods)))
    elif isinstance(obj, AgexClass):
        # Class methods
        attrs = sorted(obj.methods.keys())
    elif isinstance(obj, AgexObject):
        attrs = sorted(obj.attributes.keys())
    elif isinstance(obj, AgexModule):
        # For modules, introspect the actual registered module
        reg_module = evaluator.agent.importable_modules.get(obj.name)
        if reg_module:
            attrs = sorted([a for a in dir(reg_module.module) if not a.startswith("_")])
        else:
            attrs = []
    elif _is_bound_instance_object(obj):
        # This is a BoundInstanceObject (registered live object)
        from ..eval.objects import BoundInstanceObject

        if isinstance(obj, BoundInstanceObject):
            # Show methods and properties from the registered object
            methods = list(obj.reg_object.methods.keys())
            properties = list(obj.reg_object.properties.keys())
            attrs = sorted(methods + properties)
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
        raise AgexError("hasattr() does not take keyword arguments.")
    if len(args) != 2:
        raise AgexError(f"hasattr() takes exactly 2 arguments ({len(args)} given)")

    obj, name = args
    if not isinstance(name, str):
        raise AgexError("hasattr(): attribute name must be a string")

    # Handle AgexObjects first
    if isinstance(obj, AgexObject):
        return name in obj.attributes
    if isinstance(obj, AgexInstance):
        return name in obj.attributes or name in obj.cls.methods
    if isinstance(obj, AgexClass):
        return name in obj.methods
    if isinstance(obj, AgexModule):
        # Use JIT resolution to check if the attribute exists on the real module
        reg_module = evaluator.agent.importable_modules.get(obj.name)
        if not reg_module:
            return False
        return hasattr(reg_module.module, name) and not name.startswith("_")

    # Check for BoundInstanceObject (registered live objects)
    if _is_bound_instance_object(obj):
        from ..eval.objects import BoundInstanceObject

        if isinstance(obj, BoundInstanceObject):
            return name in obj.reg_object.methods or name in obj.reg_object.properties

    # Check registered classes and native types
    allowed = get_allowed_attributes_for_instance(evaluator.agent, obj)
    return name in allowed


def _get_general_help_text(agent: "BaseAgent") -> str:
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

    # Objects (live objects)
    if agent.object_registry:
        if parts:
            parts.append("")  # Add a blank line for separation
        parts.append("Objects:")
        for name in sorted(agent.object_registry.keys()):
            parts.append(f"  - {name}")

    if not parts:
        return (
            "No functions, classes, modules, or objects are registered with the agent."
        )

    return "Available items:\n" + "\n".join(parts)


def _format_user_function_sig(fn: UserFunction) -> str:
    """Creates a string signature for a UserFunction."""
    # This is a simplified formatter. A real one would handle more arg types.
    arg_names = [arg.arg for arg in fn.args.args]
    return f"{fn.name}({', '.join(arg_names)})"


def _help(evaluator: BaseEvaluator, *args, **kwargs) -> None:
    """Implementation of the help() builtin."""
    if kwargs:
        raise AgexError("help() does not take keyword arguments.")
    if len(args) > 1:
        raise AgexError(f"help() takes at most 1 argument ({len(args)} given)")

    obj = args[0] if args else None

    doc = None
    if obj is None:
        doc = _get_general_help_text(evaluator.agent)
    elif isinstance(obj, AgexInstance):
        # For an instance, show help for its class.
        return _help(evaluator, obj.cls)
    elif isinstance(obj, AgexClass):
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
    elif isinstance(obj, AgexModule):
        # Special handling to render help for a AgexModule
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
    elif _is_bound_instance_object(obj):
        # Handle BoundInstanceObject (registered live objects)
        from ..eval.objects import BoundInstanceObject

        if isinstance(obj, BoundInstanceObject):
            parts = [f"Help on object {obj.reg_object.name}:\n"]

            # Show methods
            methods = sorted(obj.reg_object.methods.keys())
            if methods:
                parts.append("METHODS")
                for method_name in methods:
                    method_spec = obj.reg_object.methods[method_name]
                    docstring = method_spec.docstring
                    if docstring:
                        parts.append(f"    {method_name} - {docstring}")
                    else:
                        parts.append(f"    {method_name}")

            # Show properties
            properties = sorted(obj.reg_object.properties.keys())
            if properties:
                if methods:  # Add spacing if we already showed methods
                    parts.append("")
                parts.append("PROPERTIES")
                for prop_name in properties:
                    prop_spec = obj.reg_object.properties[prop_name]
                    docstring = prop_spec.docstring
                    if docstring:
                        parts.append(f"    {prop_name} - {docstring}")
                    else:
                        parts.append(f"    {prop_name}")

            doc = "\n".join(parts)
        else:
            doc = "No help available."
    else:
        # For everything else (AgexObject, NativeFunction, raw Python objects/functions),
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
    "str": _AgexTypePlaceholder(str),
    "int": _AgexTypePlaceholder(int),
    "float": _AgexTypePlaceholder(float),
    "bool": _AgexTypePlaceholder(bool),
    "dict": _AgexTypePlaceholder(dict),
    "set": _AgexTypePlaceholder(set),
    "tuple": _AgexTypePlaceholder(tuple),
    "list": _AgexTypePlaceholder(list),
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
    "isinstance": _agex_isinstance,
    "type": _agex_type,
    # Dataclasses
    "dataclass": dataclass,
    # User-level exceptions, mapped from Python's names
    "Exception": AgexError,
    "ValueError": AgexValueError,
    "TypeError": AgexTypeError,
    "KeyError": AgexKeyError,
    "IndexError": AgexIndexError,
    "ZeroDivisionError": AgexZeroDivisionError,
    "ArithmeticError": AgexArithmeticError,
    # Agent exit signals
    "exit": ExitSuccess,
    "exit_success": ExitSuccess,
    "exit_fail": ExitFail,
    "exit_clarify": ExitClarify,
}
