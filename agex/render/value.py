from dataclasses import fields, is_dataclass
from typing import Any

from ..eval.functions import UserFunction
from ..eval.objects import AgexClass, AgexInstance, AgexObject, PrintTuple


class ValueRenderer:
    """Renders any Python value into a string suitable for an LLM prompt."""

    def __init__(self, max_len: int = 2048, max_depth: int = 2, max_items: int = 50):
        self.max_len = max_len
        self.max_depth = max_depth
        self.max_items = max_items

    def render(self, value: Any, current_depth: int = 0, compact: bool = False) -> str:
        """
        Renders a value to a string, dispatching to type-specific helpers.

        Args:
            value: The value to render
            current_depth: Current nesting depth
            compact: If True, use compact representations suitable for inline display
        """
        # Handle custom types first
        if isinstance(value, UserFunction):
            return self._render_user_function(value)
        if isinstance(value, AgexObject):
            return self._render_agex_instance_or_object(value, current_depth, compact)
        if isinstance(value, PrintTuple):
            return self._render_print_tuple(value, current_depth, compact)
        if isinstance(value, AgexInstance):
            return self._render_agex_instance_or_object(value, current_depth, compact)
        if isinstance(value, AgexClass):
            return self._render_agex_class(value)
        if is_dataclass(value) and not isinstance(value, type):
            return self._render_dataclass(value, current_depth)

        # Then primitives
        if isinstance(value, (int, float, bool, type(None))):
            return repr(value)
        if isinstance(value, str):
            return self._render_string(value)

        # Handle container truncation at max depth
        if current_depth >= self.max_depth:
            if isinstance(value, list):
                return f"[... ({len(value)} items)]"
            if isinstance(value, dict):
                return f"{{... ({len(value)} items)}}"
            if isinstance(value, set):
                return f"{{... ({len(value)} items)}}"
            if isinstance(value, AgexObject):
                return f"<{value.cls.name} object>"
            return "<...>"

        # Then recursively render containers
        if isinstance(value, list):
            return self._render_list(value, current_depth, compact)
        if isinstance(value, dict):
            return self._render_dict(value, current_depth, compact)
        if isinstance(value, set):
            return self._render_set(value, current_depth, compact)
        if isinstance(value, tuple):
            return self._render_tuple(value, current_depth, compact)

        # Fallback for all other object types
        return self._render_opaque(value, compact)

    def _render_string(self, value: str) -> str:
        if len(value) > self.max_len:
            return repr(value[: self.max_len] + "...")
        return repr(value)

    def _render_list(self, value: list, depth: int, compact: bool) -> str:
        if len(value) > self.max_items:
            return f"[... ({len(value)} items)]"
        items = []
        for item in value:
            rendered_item = self.render(item, depth + 1, compact)
            # Check length to avoid making the overall string too long
            if len(str(items)) + len(rendered_item) > self.max_len:
                items.append(f"... ({len(value) - len(items)} more)")
                break
            items.append(rendered_item)
        return f"[{', '.join(items)}]"

    def _render_dict(self, value: dict, depth: int, compact: bool) -> str:
        if len(value) > self.max_items:
            return f"{{... ({len(value)} items)}}"
        items = []
        for k, v in value.items():
            rendered_key = self.render(k, depth + 1, compact)
            rendered_value = self.render(v, depth + 1, compact)
            item_str = f"{rendered_key}: {rendered_value}"
            if len(str(items)) + len(item_str) > self.max_len:
                items.append(f"... ({len(value) - len(items)} more)")
                break
            items.append(item_str)
        return f"{{{', '.join(items)}}}"

    def _render_set(self, value: set, depth: int, compact: bool) -> str:
        if len(value) > self.max_items:
            return f"{{... ({len(value)} items)}}"
        # Very similar to list rendering
        items = []
        for item in value:
            rendered_item = self.render(item, depth + 1, compact)
            if len(str(items)) + len(rendered_item) > self.max_len:
                items.append(f"... ({len(value) - len(items)} more)")
                break
            items.append(rendered_item)
        return f"{{{', '.join(items)}}}"

    def _render_tuple(self, value: tuple, depth: int, compact: bool) -> str:
        # Tuples are immutable, but rendering is the same as list
        rendered_list = self._render_list(list(value), depth, compact)
        return f"({rendered_list[1:-1]})"

    def _render_user_function(self, value: UserFunction) -> str:
        return f"<function {value.name}>"

    def _render_agex_instance_or_object(
        self, value: Any, depth: int, compact: bool
    ) -> str:
        items = []
        for k, v in value.attributes.items():
            rendered_value = self.render(v, depth + 1, compact)
            item_str = f"{k}={rendered_value}"
            if len(str(items)) + len(item_str) > self.max_len:
                items.append("...")
                break
            items.append(item_str)
        return f"{value.cls.name}({', '.join(items)})"

    def _render_agex_class(self, value: AgexClass) -> str:
        return f"<class '{value.name}'>"

    def _render_print_tuple(self, value: PrintTuple, depth: int, compact: bool) -> str:
        """Renders the content of a PrintTuple space-separated."""
        # This rendering ignores max_len for now, as it's for a single line.
        items = [self.render(item, depth + 1, compact) for item in value]
        return " ".join(items)

    def _render_opaque(self, value: Any, compact: bool) -> str:
        type_name = type(value).__name__

        # In compact mode, prefer structural information over full string representations
        if compact:
            # For objects with shape (like numpy arrays, pandas DataFrames), show shape info
            if hasattr(value, "shape"):
                try:
                    shape_value = getattr(value, "shape", None)
                    if shape_value is not None:
                        # For DataFrames, also include column info if available
                        if hasattr(value, "columns"):
                            try:
                                columns = getattr(value, "columns", None)
                                if columns is not None:
                                    column_list = list(columns)
                                    if len(column_list) <= 10:
                                        # Show all columns if there aren't too many
                                        columns_str = str(column_list)
                                    else:
                                        # Show first 8 columns with ellipsis for larger sets
                                        columns_str = (
                                            str(column_list[:8])
                                            + f" + {len(column_list) - 8} more"
                                        )
                                    return f"<{type_name} shape={shape_value} columns={columns_str}>"
                            except Exception:
                                pass
                        return f"<{type_name} shape={shape_value}>"
                except Exception:
                    pass

            # For objects with length, show length info
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                try:
                    length = len(value)
                    return f"<{type_name} len={length}>"
                except Exception:
                    pass

            # For other objects in compact mode, use generic representation
            return f"<{type_name} object>"

        # Non-compact mode: try to get a useful string representation first
        try:
            str_repr = str(value)

            # Check if the string representation is actually informative
            # (not just the default object representation like "<object at 0x...>")
            is_default_repr = (
                (str_repr.startswith(f"<{type_name.lower()}") and "at 0x" in str_repr)
                or (
                    str_repr.startswith(f"<__main__.{type_name}")
                    and "at 0x" in str_repr
                )
                or (
                    # Handle nested class names like <test_module.test_function.<locals>.MyClass object at 0x...>
                    ".<locals>." in str_repr
                    and f".{type_name} object at 0x" in str_repr
                )
                or (
                    # Handle other module-qualified names with memory addresses
                    f"{type_name} object at 0x" in str_repr
                )
            )

            if str_repr and not is_default_repr:
                # For pandas objects and other types with useful __str__, be more permissive
                is_pandas_like = any(
                    word in type_name.lower()
                    for word in ["index", "series", "dataframe", "array"]
                )

                # Clean up multiline representations by taking first meaningful line
                if "\n" in str_repr:
                    # For pandas Index objects, join lines to preserve column names
                    if is_pandas_like and "Index([" in str_repr:
                        # Remove extra whitespace and join lines to get something like:
                        # "Index(['col1', 'col2', 'col3'], dtype='object')"
                        clean_str = " ".join(
                            line.strip()
                            for line in str_repr.split("\n")
                            if line.strip()
                        )
                        str_repr = clean_str
                    else:
                        # For other multiline objects, take first meaningful line
                        lines = str_repr.split("\n")
                        for line in lines:
                            line = line.strip()
                            if (
                                line
                                and not line.startswith("Name:")
                                and not line.startswith("dtype:")
                                and not line.startswith("Length:")
                            ):
                                str_repr = line
                                break
                        else:
                            # If no good line found, use the first non-empty line
                            str_repr = next(
                                (line.strip() for line in lines if line.strip()),
                                str_repr,
                            )

                # For short enough representations, return as-is
                if len(str_repr) <= self.max_len:
                    return str_repr

                # For longer but still useful representations, truncate intelligently
                if len(str_repr) > self.max_len:
                    # For pandas-like objects, try to preserve the structure
                    if is_pandas_like and ("[" in str_repr or "(" in str_repr):
                        # Try to truncate while preserving structure like "Index(['col1', 'col2', ...])"
                        if str_repr.endswith(")") and "(" in str_repr:
                            prefix = str_repr[: str_repr.find("(") + 1]
                            if (
                                len(prefix) < self.max_len - 10
                            ):  # Leave room for "...])"
                                return prefix + "...])"
                        elif str_repr.endswith("]") and "[" in str_repr:
                            prefix = str_repr[: str_repr.find("[") + 1]
                            if len(prefix) < self.max_len - 10:  # Leave room for "...]"
                                return prefix + "...]"

                    # For other long useful representations, truncate with ellipsis
                    return str_repr[: self.max_len - 3] + "..."

            # Fall back to informative representation without memory addresses
            # Check if the object has a useful __len__ method
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                try:
                    length = len(value)
                    return f"<{type_name} len={length}>"
                except Exception:
                    # If len() fails, continue to other checks
                    pass

            # Check if the object has a useful shape attribute
            if hasattr(value, "shape"):
                try:
                    shape_value = getattr(value, "shape", None)
                    if shape_value is not None:
                        return f"<{type_name} shape={shape_value}>"
                except Exception:
                    # If shape access fails, continue to generic fallback
                    pass

            # Generic fallback
            return f"<{type_name} object>"

        except Exception:
            # If str() fails, fall back to generic representation
            return f"<{type_name} object>"

    def _render_dataclass(self, value: Any, depth: int) -> str:
        items = []
        for f in fields(value):
            field_value = getattr(value, f.name)
            # Use compact mode for field values to get concise representations
            rendered_value = self.render(field_value, depth + 1, compact=True)
            item_str = f"{f.name}={rendered_value}"
            if len(str(items)) + len(item_str) > self.max_len:
                items.append("...")
                break
            items.append(item_str)
        return f"{type(value).__name__}({', '.join(items)})"
