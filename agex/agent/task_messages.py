"""
Task message building for agent communication.

This module handles the construction of initial task messages that agents
receive, including smart rendering of input parameters and task descriptions.
"""

import inspect
from typing import Any

from agex.render.value import ValueRenderer


def build_task_message(
    docstring: str | None,
    inputs_dataclass: type,
    inputs_instance: Any,
    return_type: type,
) -> str:
    """Build the initial user message with task description."""
    parts = []

    # Add task description
    if docstring:
        parts.append(f"Task: {docstring}")
    else:
        parts.append("Please complete the assigned task.")

    # Add note about inputs if they exist
    if inputs_instance is not None:
        parts.append(
            "Details for your task are available in the `inputs` variable. "
            "Here is its structure and content:"
        )

        # Render each input field individually using smart rendering
        input_parts = []
        example_field: str
        for field in inputs_dataclass.__dataclass_fields__.values():
            field_value = getattr(inputs_instance, field.name)
            smart_rendered = _smart_render_for_task_input(field_value)
            input_parts.append(f"inputs.{field.name} = {smart_rendered}")
            example_field = field.name

        # Create the full inputs display
        inputs_content = "\n".join(input_parts)
        parts.append(f"```\n{inputs_content}\n```")
        if example_field:
            parts.append(
                f"\nAccess these values with patterns like `inputs.{example_field}`\n"
            )

    # Add expected output format with clarification for function types
    if return_type is inspect.Parameter.empty:
        # No return type annotation - just call task_success() with no arguments
        parts.append("When complete, call `task_success()` to indicate completion.")
    elif "Callable" in str(return_type):
        # Function return type - special instructions
        # Clean up the type representation to remove confusing module references
        return_type_str = str(return_type)
        # Remove "typing." prefix but keep the useful type information
        if return_type_str.startswith("typing."):
            return_type_str = return_type_str[7:]  # Remove "typing." prefix

        parts.append(
            f"When complete, call `task_success(your_function)` where your_function is the {return_type_str} you created. "
            "Pass the function object itself, not the result of calling the function.\n"
        )
    else:
        # Regular return type - show the type annotation
        # Use clean type names for all types when possible
        if (
            hasattr(return_type, "__module__")
            and hasattr(return_type, "__name__")
            and not hasattr(return_type, "__origin__")  # Not a generic type
        ):
            # Use the clean class name for simple types (str, int, custom classes)
            return_type_name = return_type.__name__
        else:
            # For generic types (list[int], dict[str, int]) or complex types,
            # use the full string representation to preserve type parameters
            return_type_name = str(return_type)

        parts.append(
            f"When complete, call `task_success(result)` with your result. The result type should be `{return_type_name}`."
        )

    return "\n\n".join(parts)


def _smart_render_for_task_input(value: Any) -> str:
    """
    Smart rendering for individual task input values.

    Uses ValueRenderer with generous task-appropriate limits to show
    rich content like DataFrames, arrays, and other complex objects
    in their natural representation when possible.
    """
    renderer = ValueRenderer(
        max_len=4096,  # Generous limit for rich task display
        max_depth=4,  # Deep enough for complex nested objects
        max_items=50,  # Show more items than default for task context
    )
    return renderer.render(value, compact=False)
