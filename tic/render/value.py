from typing import Any

from ..eval.call import PrintTuple
from ..eval.functions import UserFunction
from ..eval.objects import TicObject


class ValueRenderer:
    """Renders any Python value into a string suitable for an LLM prompt."""

    def __init__(self, max_len: int = 128, max_depth: int = 2, max_items: int = 50):
        self.max_len = max_len
        self.max_depth = max_depth
        self.max_items = max_items

    def render(self, value: Any, current_depth: int = 0) -> str:
        """
        Renders a value to a string, dispatching to type-specific helpers.
        """
        # Handle custom types first
        if isinstance(value, UserFunction):
            return self._render_user_function(value)
        if isinstance(value, TicObject):
            return self._render_tic_object(value, current_depth)
        if isinstance(value, PrintTuple):
            return self._render_print_tuple(value, current_depth)

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
            if isinstance(value, TicObject):
                return f"<{value.cls.name} object>"
            return "<...>"

        # Then recursively render containers
        if isinstance(value, list):
            return self._render_list(value, current_depth)
        if isinstance(value, dict):
            return self._render_dict(value, current_depth)
        if isinstance(value, set):
            return self._render_set(value, current_depth)
        if isinstance(value, tuple):
            return self._render_tuple(value, current_depth)

        # Fallback for all other object types
        return self._render_opaque(value)

    def _render_string(self, value: str) -> str:
        if len(value) > self.max_len:
            return repr(value[: self.max_len] + "...")
        return repr(value)

    def _render_list(self, value: list, depth: int) -> str:
        if len(value) > self.max_items:
            return f"[... ({len(value)} items)]"
        items = []
        for item in value:
            rendered_item = self.render(item, depth + 1)
            # Check length to avoid making the overall string too long
            if len(str(items)) + len(rendered_item) > self.max_len:
                items.append(f"... ({len(value) - len(items)} more)")
                break
            items.append(rendered_item)
        return f"[{', '.join(items)}]"

    def _render_dict(self, value: dict, depth: int) -> str:
        if len(value) > self.max_items:
            return f"{{... ({len(value)} items)}}"
        items = []
        for k, v in value.items():
            rendered_key = self.render(k, depth + 1)
            rendered_value = self.render(v, depth + 1)
            item_str = f"{rendered_key}: {rendered_value}"
            if len(str(items)) + len(item_str) > self.max_len:
                items.append(f"... ({len(value) - len(items)} more)")
                break
            items.append(item_str)
        return f"{{{', '.join(items)}}}"

    def _render_set(self, value: set, depth: int) -> str:
        if len(value) > self.max_items:
            return f"{{... ({len(value)} items)}}"
        # Very similar to list rendering
        items = []
        for item in value:
            rendered_item = self.render(item, depth + 1)
            if len(str(items)) + len(rendered_item) > self.max_len:
                items.append(f"... ({len(value) - len(items)} more)")
                break
            items.append(rendered_item)
        return f"{{{', '.join(items)}}}"

    def _render_tuple(self, value: tuple, depth: int) -> str:
        # Tuples are immutable, but rendering is the same as list
        rendered_list = self._render_list(list(value), depth)
        return f"({rendered_list[1:-1]})"

    def _render_user_function(self, value: UserFunction) -> str:
        return f"<function {value.name}>"

    def _render_tic_object(self, value: TicObject, depth: int) -> str:
        items = []
        for k, v in value.attributes.items():
            rendered_value = self.render(v, depth + 1)
            item_str = f"{k}={rendered_value}"
            if len(str(items)) + len(item_str) > self.max_len:
                items.append("...")
                break
            items.append(item_str)
        return f"{value.cls.name}({', '.join(items)})"

    def _render_print_tuple(self, value: PrintTuple, depth: int) -> str:
        """Renders the content of a PrintTuple space-separated."""
        # This rendering ignores max_len for now, as it's for a single line.
        items = [self.render(item, depth + 1) for item in value]
        return " ".join(items)

    def _render_opaque(self, value: Any) -> str:
        type_name = type(value).__name__
        if hasattr(value, "shape"):
            return f"<{type_name} shape={value.shape}>"
        if hasattr(value, "__len__"):
            return f"<{type_name} len={len(value)}>"
        return f"<{type_name} object>"
