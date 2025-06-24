from typing import Any

from tic.agent import Agent
from tic.eval.functions import NativeFunction
from tic.eval.objects import TicModule


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


def help_builtin(obj: Any = None, *, agent: "Agent") -> str:
    """Provides help information about a registered object or lists all available items."""
    if obj is None:
        return _get_general_help_text(agent)

    doc = None
    if isinstance(obj, NativeFunction):
        doc = obj.fn.__doc__
    elif isinstance(obj, TicModule):
        # Special handling to render help for a TicModule
        parts = ["Help on module " + obj.__name__ + ":\n"]
        # Introspect the module for contents
        contents = sorted([attr for attr in dir(obj) if not attr.startswith("_")])
        if contents:
            parts.append("CONTENTS")
            parts.extend([f"    {item}" for item in contents])
        return "\n".join(parts)
    else:
        doc = getattr(obj, "__doc__", None)

    if doc:
        # A simple docstring renderer for now.
        name = (
            obj.name
            if isinstance(obj, NativeFunction)
            else getattr(obj, "__name__", "object")
        )
        return f"Help for {name}:\n\n{doc.strip()}"

    # Fallback for now
    # TODO: Implement help for specific objects
    return f"Help is not yet available for objects of type: {type(obj).__name__}"
