from typing import Any, List

from ..tokenizers import Tokenizer, get_tokenizer
from .value import ValueRenderer


class StreamRenderer:
    """
    Renders streams of Python objects into strings, respecting a token budget.
    This class is responsible for the low-level rendering logic.
    """

    def __init__(self, model_name: str):
        self.tokenizer: Tokenizer = get_tokenizer(model_name)
        self.value_renderer = ValueRenderer()

    def render_state_stream(self, items: dict[str, Any], budget: int) -> str:
        """Renders state changes with degradation logic."""
        output_lines: List[str] = []
        remaining_budget = budget
        omitted_items = False

        for key, value in reversed(list(items.items())):
            # Attempt to render with default detail.
            rendered_line, cost, success = self._render_and_check(
                key, value, remaining_budget, depth=2
            )
            # If it fails, try a more summarized version.
            if not success:
                rendered_line, cost, success = self._render_and_check(
                    key, value, remaining_budget, depth=0
                )

            if success:
                if rendered_line:
                    output_lines.insert(0, rendered_line)
                remaining_budget -= cost
            else:
                omitted_items = True

        if omitted_items and output_lines:
            marker = "..."
            marker_cost = len(self.tokenizer.encode(marker + "\n"))
            if remaining_budget >= marker_cost:
                output_lines.insert(0, marker)

        return "\n".join(output_lines)

    def render_item_stream(
        self,
        items: List[Any],
        budget: int,
        header: str = "",
    ) -> str:
        """
        Renders a generic stream of items, keeping the most recent ones that fit
        within the budget and adding a truncation marker if necessary.
        """
        if not items:
            return ""

        render_func = self.value_renderer.render
        header_cost = len(self.tokenizer.encode(header))
        lines_to_render = []
        current_cost = 0

        for item in reversed(items):
            rendered_line = render_func(item)
            line_cost = len(self.tokenizer.encode(rendered_line + "\n"))

            if header_cost + current_cost + line_cost > budget:
                if lines_to_render:
                    marker = "...\n"
                    marker_cost = len(self.tokenizer.encode(marker))
                    if header_cost + current_cost + marker_cost <= budget:
                        lines_to_render.insert(0, "...")
                break

            lines_to_render.insert(0, rendered_line)
            current_cost += line_cost

        if not lines_to_render:
            return ""

        return header + "\n".join(lines_to_render)

    def _render_and_check(
        self, key: str, value: Any, budget: int, depth: int
    ) -> tuple[str, int, bool]:
        """Helper to centralize the render -> tokenize -> check loop."""
        original_max_len = self.value_renderer.max_len
        if depth == 0:
            self.value_renderer.max_len = 32  # Force very short strings for summary

        self.value_renderer.max_depth = depth
        rendered_value = self.value_renderer.render(value)
        self.value_renderer.max_len = original_max_len  # Restore

        line = f"{key} = {rendered_value}"
        # Add a newline for accurate token counting of multi-line context
        cost = len(self.tokenizer.encode(line + "\n"))

        if cost <= budget:
            return line, cost, True
        else:
            return "", 0, False
