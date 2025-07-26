from dataclasses import fields, is_dataclass
from typing import Any

from ..eval.functions import UserFunction
from ..eval.objects import AgexClass, AgexInstance, AgexObject, PrintAction


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
        if isinstance(value, PrintAction):
            return self._render_print_action(value, current_depth, compact)
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

    def _render_print_action(
        self, value: PrintAction, depth: int, compact: bool
    ) -> str:
        """Renders the content of a PrintAction space-separated."""
        # This rendering ignores max_len for now, as it's for a single line.
        items = [self.render(item, depth + 1, compact) for item in value]
        return " ".join(items)

    def _render_opaque(self, value: Any, compact: bool) -> str:
        type_name = type(value).__name__

        if compact:
            return self._render_compact_metadata(value, type_name)

        # Try to get natural string representation
        try:
            str_repr = str(value)
        except Exception:
            return self._render_metadata_fallback(value, type_name)

        # Check if it's actually informative (not default object repr)
        if self._is_default_object_repr(str_repr, type_name):
            return self._render_metadata_fallback(value, type_name)

        # Apply intelligent truncation if needed
        if len(str_repr) <= self.max_len:
            return str_repr

        return self._truncate_intelligently(str_repr)

    def _render_compact_metadata(self, value: Any, type_name: str) -> str:
        """Render compact metadata using introspection."""
        metadata = []

        # Check for shape (arrays, dataframes, etc.)
        if hasattr(value, "shape"):
            try:
                shape = getattr(value, "shape")
                if shape is not None:
                    metadata.append(f"shape={shape}")
            except Exception:
                pass

        # Check for length
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            try:
                length = len(value)
                metadata.append(f"len={length}")
            except Exception:
                pass

        # Check for columns (dataframes, etc.)
        if hasattr(value, "columns"):
            try:
                columns = list(getattr(value, "columns"))
                if len(columns) <= 5:
                    metadata.append(f"columns={columns}")
                else:
                    metadata.append(f"columns=[{len(columns)} cols]")
            except Exception:
                pass

        if metadata:
            return f"<{type_name} {' '.join(metadata)}>"
        return f"<{type_name} object>"

    def _is_default_object_repr(self, str_repr: str, type_name: str) -> bool:
        """Check if string representation is the default object repr."""
        import re

        # Pattern: <ClassName object at 0x...> or similar
        default_patterns = [
            rf"<{re.escape(type_name.lower())}.*at 0x",
            rf"<.*\.{re.escape(type_name)}.*at 0x",
            rf"<.*{re.escape(type_name)} object at 0x",
        ]

        for pattern in default_patterns:
            if re.search(pattern, str_repr, re.IGNORECASE):
                return True
        return False

    def _render_metadata_fallback(self, value: Any, type_name: str) -> str:
        """Fallback when string repr is not informative."""
        return self._render_compact_metadata(value, type_name)

    def _truncate_intelligently(self, str_repr: str) -> str:
        """Apply intelligent truncation based on content patterns."""

        # For multiline content, try to preserve structure
        if "\n" in str_repr:
            return self._truncate_multiline(str_repr)

        # For single line, simple truncation with ellipsis
        return str_repr[: self.max_len - 3] + "..."

    def _truncate_multiline(self, str_repr: str) -> str:
        """Intelligently truncate multiline content."""
        lines = str_repr.split("\n")

        # If it's short enough, keep as-is
        if len(str_repr) <= self.max_len:
            return str_repr

        # Detect if it looks like tabular data (aligned columns)
        if self._looks_like_table(lines):
            return self._truncate_table(lines)

        # Detect if it looks like a list/array
        if self._looks_like_list(str_repr):
            return self._truncate_list_like(str_repr)

        # For other multiline content, try to keep first few meaningful lines
        return self._truncate_generic_multiline(lines)

    def _looks_like_table(self, lines: list[str]) -> bool:
        """Detect if content looks like tabular data."""
        if len(lines) < 2:
            return False

        # Look for consistent spacing/alignment patterns
        # Tables often have headers followed by data rows with similar structure
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) < 2:
            return False

        # Check if lines have similar lengths (suggesting columns)
        lengths = [len(line.rstrip()) for line in non_empty_lines[:5]]
        if len(set(lengths)) <= 2:  # Similar lengths suggest table structure
            return True

        # Check for numeric patterns that suggest data rows
        import re

        data_line_pattern = re.compile(r"^\s*\d+\s+.*\d+")
        data_lines = [
            line for line in non_empty_lines[1:6] if data_line_pattern.match(line)
        ]
        return len(data_lines) >= 2

    def _looks_like_list(self, str_repr: str) -> bool:
        """Detect if content looks like a list or array."""
        import re

        # Look for list-like patterns: [...], array([...]), etc.
        list_patterns = [
            r"^\s*\[.*\]\s*$",  # [item1, item2, ...]
            r"^\s*array\(\[.*\]\)\s*$",  # array([...])
            r"^\s*{\s*.*\s*}\s*$",  # {item1, item2, ...}
        ]

        for pattern in list_patterns:
            if re.match(pattern, str_repr, re.DOTALL):
                return True
        return False

    def _truncate_table(self, lines: list[str]) -> str:
        """Truncate table-like content, preserving headers and some data."""
        budget = self.max_len
        result_lines = []
        current_len = 0

        # Always try to include the first line (likely headers)
        if lines and lines[0].strip():
            first_line = lines[0] + "\n"
            if len(first_line) < budget:
                result_lines.append(lines[0])
                current_len += len(first_line)

        # Add data lines until we run out of budget
        for line in lines[1:]:
            line_with_newline = line + "\n"
            if (
                current_len + len(line_with_newline) > budget - 10
            ):  # Leave room for "..."
                break
            result_lines.append(line)
            current_len += len(line_with_newline)

        # Add ellipsis if we truncated
        if len(result_lines) < len(lines):
            result_lines.append("...")

        return "\n".join(result_lines)

    def _truncate_list_like(self, str_repr: str) -> str:
        """Truncate list-like content, preserving structure."""
        # Try to preserve opening and closing brackets/parens
        if len(str_repr) <= self.max_len:
            return str_repr

        # Find the opening structure
        for i, char in enumerate(str_repr):
            if char in "([{":
                # Find a good breaking point
                truncate_point = self.max_len - 10  # Leave room for "...]" or similar
                if truncate_point > i:
                    prefix = str_repr[:truncate_point]
                    # Try to end at a reasonable boundary (comma, space)
                    for boundary in [", ", " ", ","]:
                        if boundary in prefix[i:]:
                            last_boundary = prefix.rfind(boundary)
                            if last_boundary > i:
                                prefix = prefix[:last_boundary]
                                break

                    # Add appropriate closing
                    closing = {"(": "...)", "[": "...]", "{": "...}"}
                    return prefix + closing.get(char, "...")

        # Fallback to simple truncation
        return str_repr[: self.max_len - 3] + "..."

    def _truncate_generic_multiline(self, lines: list[str]) -> str:
        """Truncate generic multiline content."""
        budget = self.max_len
        result_lines = []
        current_len = 0

        for line in lines:
            line_with_newline = line + "\n"
            if current_len + len(line_with_newline) > budget - 10:
                break

            # Skip metadata-like lines for generic content
            if line.strip() and not any(
                line.strip().startswith(prefix)
                for prefix in ["Name:", "dtype:", "Length:", "Type:"]
            ):
                result_lines.append(line)
                current_len += len(line_with_newline)

        if len(result_lines) < len(lines):
            result_lines.append("...")

        return "\n".join(result_lines)

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
