from typing import Any

from tic.agent import Agent


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

    # Fallback for now
    # TODO: Implement help for specific objects
    return f"Help is not yet available for objects of type: {type(obj).__name__}"
