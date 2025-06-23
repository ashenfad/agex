from typing import Any, Callable, List, Tuple

from ..state import Versioned
from ..tokenizers import Tokenizer, get_tokenizer
from .value import ValueRenderer


class ContextRenderer:
    """
    Renders the current agent context (recent state changes and print output)
    into a string suitable for an LLM prompt, respecting a token budget.
    """

    def __init__(self, model_name: str):
        self.tokenizer: Tokenizer = get_tokenizer(model_name)
        self.value_renderer = ValueRenderer()

    def render(self, state: Versioned, budget: int) -> str:
        """
        Orchestrates the rendering of the current context.

        It prioritizes the most recent changes and gracefully degrades
        the level of detail to fit within the token budget.
        """
        # 1. Separate state changes from stdout content.
        stdout_content = state.ephemeral.get("__stdout__", [])
        state_changes = [
            (k, v) for k, v in state.ephemeral.items() if k != "__stdout__"
        ]

        # 2. Allocate independent budgets.
        stdout_budget = int(budget * 0.4)
        state_budget = int(budget * 0.6)

        # 3. Render each stream independently.
        rendered_stdout = self._render_item_stream(
            items=stdout_content,
            budget=stdout_budget,
            render_func=self.value_renderer.render,
            header="Agent printed:\n",
        )

        rendered_state = self._render_state_stream(
            items=state_changes, budget=state_budget
        )

        # 4. Combine the rendered outputs.
        return "\n".join(filter(None, [rendered_state, rendered_stdout]))

    def _render_state_stream(self, items: List[Tuple[str, Any]], budget: int) -> str:
        """Renders state changes with degradation logic."""
        output_lines: List[str] = []
        remaining_budget = budget

        for key, value in reversed(items):
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

        return "\n".join(output_lines)

    def _render_item_stream(
        self,
        items: List[Any],
        budget: int,
        render_func: Callable[[Any], str],
        header: str = "",
    ) -> str:
        """
        Renders a generic stream of items, keeping the most recent ones that fit
        within the budget and adding a truncation marker if necessary.
        """
        if not items:
            return ""

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
        self.value_renderer.max_depth = depth
        rendered_value = self.value_renderer.render(value)

        line = f"{key} = {rendered_value}"
        # Add a newline for accurate token counting of multi-line context
        cost = len(self.tokenizer.encode(line + "\n"))

        if cost <= budget:
            return line, cost, True
        else:
            return "", 0, False
