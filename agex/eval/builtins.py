import copy
import inspect
from typing import Any

from kvit import Store

from agex.agent.base import BaseAgent
from agex.agent.events import OutputEvent
from agex.eval.objects import (
    AgexClass,
    AgexInstance,
    AgexModule,
    BoundInstanceObject,
    ImageAction,
)
from agex.eval.user_errors import AgexValueError
from agex.state import is_live_root


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


def _view_image_stateful(
    image: Any, detail: str = "high", *, state: Store, agent_name: str, on_event=None
) -> None:
    """
    A custom builtin to "view" an image, which adds an ImageAction to the event log.
    """
    if detail not in ("low", "high"):
        raise AgexValueError("detail must be 'low' or 'high'")

    # "Snapshot" the arguments to ensure immutability in the log
    is_live = is_live_root(state)
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


def _format_user_function_sig(fn) -> str:
    """Formats a UserFunction into a signature string."""
    # This is a simplified formatter. A real one would handle more arg types.
    arg_names = [arg.arg for arg in fn.args.args]
    return f"{fn.name}({', '.join(arg_names)})"


def _get_general_help_text(agent: "BaseAgent") -> str:
    """Returns a string with a summary of all registered items."""
    parts = ["Available items:"]

    # Functions and classes from policy __main__
    try:
        main_ns = agent._policy.namespaces.get("__main__")
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
        for name, ns in agent._policy.namespaces.items():
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
        ns = agent._policy.namespaces.get(item.name)
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
