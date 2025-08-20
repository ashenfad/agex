import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.agent.datatypes import (
    TaskClarify,
    TaskContinue,
    TaskFail,
    TaskSuccess,
)
from agex.agent.events import OutputEvent
from agex.eval.objects import (
    AgexClass,
    AgexDataClass,
    AgexInstance,
    AgexModule,
    AgexObject,
    BoundInstanceObject,
    ImageAction,
)
from agex.eval.user_errors import (
    AgexArithmeticError,
    AgexAttributeError,
    AgexError,
    AgexIndexError,
    AgexKeyError,
    AgexTypeError,
    AgexValueError,
    AgexZeroDivisionError,
)
from agex.eval.utils import get_allowed_attributes_for_instance
from agex.state import Live, State


def _smart_render_for_snapshot(value: Any) -> str:
    """
    Smart rendering for snapshotting objects in live mode.
    Uses ValueRenderer with conservative limits to avoid huge strings.
    """
    from agex.render.value import ValueRenderer

    renderer = ValueRenderer(max_len=512, max_depth=2)
    return renderer.render(value)


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


def _print_stateful(*args: Any, state: State, agent_name: str, on_event=None):
    """
    A custom implementation of 'print' that appends its arguments to the
    `__event_log__` list in the agent's state as a single `OutputEvent`.
    This function is "store-aware" to ensure the event log is immutable.
    """
    snapped_args: tuple
    try:
        snapped_args = copy.deepcopy(args)
    except Exception:
        # Fall back to smart rendering for both state types
        snapped_args = tuple(_smart_render_for_snapshot(arg) for arg in args)

    # Create and add the event using efficient reference-based storage
    from agex.state.log import add_event_to_log

    event = OutputEvent(agent_name=agent_name, parts=list(snapped_args))
    add_event_to_log(state, event, on_event=on_event)


def _view_image_stateful(
    image: Any, detail: str = "high", *, state: State, agent_name: str, on_event=None
) -> None:
    """
    A custom builtin to "view" an image, which adds an ImageAction to the event log.
    """
    if detail not in ("low", "high"):
        raise AgexValueError("detail must be 'low' or 'high'")

    # "Snapshot" the arguments to ensure immutability in the log
    is_live = isinstance(state.base_store, Live)
    snapped_image: Any
    try:
        if is_live:
            snapped_image = copy.deepcopy(image)
        else:
            snapped_image = image

            # Test if this would be serializable to avoid breaking the event log
            import pickle

            image_action = ImageAction(image=snapped_image, detail=detail)
            test_event = OutputEvent(agent_name=agent_name, parts=[image_action])
            pickle.dumps(test_event)  # This will raise if unpicklable

    except Exception:
        # Fall back to smart rendering for both state types
        snapped_image = _smart_render_for_snapshot(image)

    # For now, ImageAction is a dataclass that gets put inside an OutputEvent
    image_action = ImageAction(image=snapped_image, detail=detail)

    # Create and add the event using efficient reference-based storage
    from agex.state.log import add_event_to_log

    event = OutputEvent(agent_name=agent_name, parts=[image_action])
    add_event_to_log(state, event, on_event=on_event)


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


def _dir(evaluator, *args, **kwargs) -> list[str]:
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
        # Policy-backed enumeration only
        ns = evaluator.agent._policy.namespaces.get(obj.name)  # type: ignore[attr-defined]
        if ns is None:
            attrs = []
        else:
            from agex.agent.policy.describe import describe_namespace

            desc = describe_namespace(ns, include_low=False)
            attrs = sorted(k for k in desc.keys())
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

    # Scrub any private/protected attributes from the final list
    final_attrs = [attr for attr in attrs if not attr.startswith("_")]

    # No deepcopy needed here, as `attrs` is a new list of strings, which is immutable.
    # Create and add the event using efficient reference-based storage
    from agex.state.log import add_event_to_log

    event = OutputEvent(agent_name=evaluator.agent.name, parts=[final_attrs])
    add_event_to_log(evaluator.state, event)

    return final_attrs


def _hasattr(evaluator, *args, **kwargs) -> bool:
    """
    Implementation of the hasattr() builtin.
    """
    if kwargs:
        raise AgexError("hasattr() does not take keyword arguments.")
    if len(args) != 2:
        raise AgexError(f"hasattr() takes exactly 2 arguments ({len(args)} given)")

    obj, name = args
    if not isinstance(name, str):
        raise AgexError("hasattr(): attribute name must be a string")

    # Handle AgexObjects first
    if isinstance(obj, AgexModule):
        # Policy-backed check for module attributes only
        res = evaluator.agent._policy.resolve_module_member(obj.name, name)  # type: ignore[attr-defined]
        return res is not None
    if isinstance(obj, (AgexObject, AgexInstance, BoundInstanceObject)):
        try:
            obj.getattr(name)
            return True
        except AgexAttributeError:
            return False

    # For all other objects, respect the agent's sandbox rules.
    allowed = get_allowed_attributes_for_instance(evaluator.agent, obj)
    return name in allowed


def _getattr(evaluator, *args, **kwargs) -> Any:
    """
    Implementation of the getattr() builtin.
    """
    if kwargs:
        raise AgexError("getattr() does not take keyword arguments.")
    if len(args) not in [2, 3]:
        raise AgexError(f"getattr() takes 2 or 3 arguments ({len(args)} given)")

    obj, name = args[0], args[1]
    default = args[2] if len(args) == 3 else None

    if not isinstance(name, str):
        raise AgexTypeError("getattr(): attribute name must be a string")

    # For AgexObjects, use their custom getattr
    if isinstance(obj, AgexModule):
        # Policy-backed check for module attributes only
        res = evaluator.agent._policy.resolve_module_member(obj.name, name)  # type: ignore[attr-defined]
        if res is not None:
            if hasattr(res, "fn"):
                return res.fn
            return res
        if len(args) == 3:
            return default
        raise AgexAttributeError(f"'{obj.name}' module has no attribute '{name}'")
    if isinstance(obj, (AgexObject, AgexInstance, BoundInstanceObject)):
        try:
            return obj.getattr(name)
        except AgexAttributeError:
            if len(args) == 3:
                return default
            raise

    # For all other objects, respect the agent's sandbox rules.
    allowed = get_allowed_attributes_for_instance(evaluator.agent, obj)
    if name in allowed:
        if hasattr(obj, name):
            return getattr(obj, name)
    if len(args) == 3:
        return default
    # If the attribute is not allowed, raise an AttributeError.
    # This is consistent with how normal attribute access is handled.
    raise AgexAttributeError(f"'{type(obj).__name__}' object has no attribute '{name}'")


def _get_general_help_text(agent: "BaseAgent") -> str:
    """Returns a string with a summary of all registered items."""
    parts = ["Available items:"]

    # Functions and classes from policy __main__
    try:
        main_ns = agent._policy.namespaces.get("__main__")  # type: ignore[attr-defined]
    except Exception:
        main_ns = None
    if main_ns is not None and main_ns.kind == "virtual":
        fns = sorted(main_ns.fns.keys())
        if fns:
            parts.append("\nFunctions:")
            parts.extend([f"- {fn}" for fn in fns])
        clss = sorted(main_ns.classes.keys())
        if clss:
            parts.append("\nClasses:")
            parts.extend([f"- {cls}" for cls in clss])

    # Modules and objects from policy namespaces
    mods = []
    objects = []
    try:
        for name, ns in agent._policy.namespaces.items():  # type: ignore[attr-defined]
            if name == "__main__":
                continue
            if getattr(ns, "kind", None) == "module":
                mods.append(name)
            elif getattr(ns, "kind", None) == "instance":
                objects.append(name)
    except Exception:
        pass
    all_objects = sorted(set(mods) | set(objects))
    if all_objects:
        parts.append("\nObjects:")
        parts.extend([f"- {obj}" for obj in all_objects])

    if len(parts) == 1:  # Only "Available items:" was added
        return "No resources registered with the agent."

    return "\n".join(parts)


def _format_user_function_sig(fn) -> str:
    """Formats a UserFunction into a signature string."""
    # This is a simplified formatter. A real one would handle more arg types.
    arg_names = [arg.arg for arg in fn.args.args]
    return f"{fn.name}({', '.join(arg_names)})"


def _get_help_text(agent: "BaseAgent", item: Any) -> str:
    """Returns a detailed help string for a specific registered item."""
    if isinstance(item, AgexInstance):
        # For an instance, show help for its class.
        return _get_help_text(agent, item.cls)
    if isinstance(item, AgexClass):
        parts = [f"Help on class {item.name}:\n"]
        if "__init__" in item.methods:
            init_sig = _format_user_function_sig(item.methods["__init__"])
            parts.append(f"{item.name}{init_sig.replace('__init__', '', 1)}")
        else:
            parts.append(f"{item.name}()")

        methods = sorted(item.methods.keys())
        if methods:
            parts.append("\nMethods defined here:")
            for method_name in methods:
                method_sig = _format_user_function_sig(item.methods[method_name])
                parts.append(f"  {method_sig}")
        return "\n".join(parts)
    if isinstance(item, AgexModule):
        parts = ["Help on module " + item.name + ":\n"]
        ns = agent._policy.namespaces.get(item.name)  # type: ignore[attr-defined]
        if ns is not None:
            from agex.agent.policy.describe import describe_namespace

            contents = sorted(
                k
                for k in describe_namespace(ns, include_low=False).keys()
                if not k.startswith("_")
            )
            if contents:
                parts.append("CONTENTS")
                parts.extend([f"    {x}" for x in contents])
        return "\n".join(parts)
    if _is_bound_instance_object(item):
        from ..eval.objects import BoundInstanceObject

        if isinstance(item, BoundInstanceObject):
            parts = [f"Help on object {item.reg_object.name}:\n"]
            # Methods
            methods = sorted(item.reg_object.methods.keys())
            if methods:
                parts.append("METHODS")
                for name in methods:
                    doc = item.reg_object.methods[name].docstring
                    parts.append(f"    {name} - {doc}" if doc else f"    {name}")
            # Properties
            properties = sorted(item.reg_object.properties.keys())
            if properties:
                if methods:
                    parts.append("")
                parts.append("PROPERTIES")
                for name in properties:
                    doc = item.reg_object.properties[name].docstring
                    parts.append(f"    {name} - {doc}" if doc else f"    {name}")
            return "\n".join(parts)
    # For other types, try to get a docstring.
    return inspect.getdoc(item) or "No help available."


def _is_allowed_for_help(item: Any) -> bool:
    """Check if an item is allowed for help() - registered resources or basic Python types."""
    return (
        isinstance(item, (AgexClass, AgexInstance, AgexModule))
        or _is_bound_instance_object(item)
        or isinstance(item, (int, float, str, bool, list, dict, tuple, set, type(None)))
        or hasattr(item, "__doc__")  # Any object with documentation
    )


def _help(evaluator, *args, **kwargs) -> None:
    """Implementation of the help() builtin."""
    if kwargs:
        raise AgexError("help() does not take keyword arguments.")
    if len(args) > 1:
        raise AgexError(f"help() takes at most 1 argument ({len(args)} given)")

    item = args[0] if args else None

    if item is not None and not _is_allowed_for_help(item):
        raise AgexTypeError("help() is only supported for registered resources.")

    doc = (
        _get_help_text(evaluator.agent, item)
        if item
        else _get_general_help_text(evaluator.agent)
    )
    # Print the help text to stdout
    # No deepcopy needed, `doc` is a string.
    # Create and add the event using efficient reference-based storage
    from agex.state.log import add_event_to_log

    event = OutputEvent(agent_name=evaluator.agent.name, parts=[doc])
    add_event_to_log(evaluator.state, event, on_event=evaluator.on_event)


def _task_continue_with_observations(
    *observations: Any, state: State, agent_name: str, on_event=None
) -> None:
    """
    Signal to the agent to continue, providing a list of observations.
    This is effectively a programmatic `print()` that also forces a continue.
    """
    _print_stateful(
        *observations, state=state, agent_name=agent_name, on_event=on_event
    )
    raise TaskContinue()


STATEFUL_BUILTINS: dict[str, StatefulFn] = {
    "print": StatefulFn(_print_stateful),
    "view_image": StatefulFn(_view_image_stateful),
    "help": StatefulFn(_help, needs_evaluator=True),
    "dir": StatefulFn(_dir, needs_evaluator=True),
    "hasattr": StatefulFn(_hasattr, needs_evaluator=True),
    "getattr": StatefulFn(_getattr, needs_evaluator=True),
    "task_continue": StatefulFn(_task_continue_with_observations),
}


# This is the main registry of built-in functions available in the sandbox.
BUILTINS = {
    "abs": abs,
    "len": len,
    "max": max,
    "min": min,
    "sum": sum,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "bytes": bytes,
    "dict": _AgexTypePlaceholder(dict),
    "set": _AgexTypePlaceholder(set),
    "tuple": _AgexTypePlaceholder(tuple),
    "list": _AgexTypePlaceholder(list),
    "round": round,
    "pow": pow,
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
    # Task control functions
    "task_success": TaskSuccess,
    "task_fail": TaskFail,
    "task_clarify": TaskClarify,
}
