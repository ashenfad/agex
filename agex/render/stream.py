from typing import Any, List, Optional

from ..eval.objects import ImageAction, PrintAction
from ..llm.core import ContentPart, ImagePart, TextPart
from ..tokenizers import Tokenizer, get_tokenizer
from .primitives import (
    estimate_image_cost,
    get_image_error_message,
    serialize_image_to_base64,
)
from .value import ValueRenderer


class StreamRenderer:
    """
    Renders streams of Python objects into strings or multimodal content parts,
    respecting a token budget. This class is responsible for low-level rendering.
    """

    def __init__(self, model_name: str):
        self.tokenizer: Tokenizer = get_tokenizer(model_name)
        self.value_renderer = ValueRenderer(max_len=4096, max_depth=4)

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
    ) -> List[ContentPart]:
        """
        Renders a generic stream of items into a list of ContentParts, keeping
        the most recent ones that fit within the budget.
        """
        if not items:
            return []

        render_func = self.value_renderer.render
        # Store tuples of (ContentPart, cost) to manage budget.
        parts_with_cost: List[tuple[ContentPart, int]] = []
        current_cost = 0
        omitted_items = False

        for item in reversed(items):
            part: Optional[ContentPart] = None
            cost = 0

            if isinstance(item, PrintAction):
                rendered_args = [render_func(arg) for arg in item]
                rendered_line = " ".join(map(str, rendered_args))
                cost = len(self.tokenizer.encode(rendered_line + "\n"))
                part = TextPart(text=rendered_line)

            elif isinstance(item, ImageAction):
                cost = estimate_image_cost(item.image, item.detail)
                # Only serialize if it might fit.
                if current_cost + cost <= budget:
                    base64_image = serialize_image_to_base64(item.image)
                    if base64_image:
                        part = ImagePart(image=base64_image)
                    else:
                        # Generate error message based on image type
                        placeholder = get_image_error_message(item.image)
                        cost = len(self.tokenizer.encode(placeholder + "\n"))
                        part = TextPart(text=placeholder)
                else:
                    cost = 0  # Reset cost, we are not adding this part

            else:  # Fallback for other raw types in the stream
                rendered_line = render_func(item)
                cost = len(self.tokenizer.encode(rendered_line + "\n"))
                part = TextPart(text=rendered_line)

            if part and current_cost + cost <= budget:
                parts_with_cost.insert(0, (part, cost))
                current_cost += cost
            elif cost > 0:  # If we calculated a cost but didn't add the part
                omitted_items = True

        # Post-processing: add truncation markers
        final_parts: List[ContentPart] = [p for p, c in parts_with_cost]

        if omitted_items and final_parts:
            marker = "..."
            marker_cost = len(self.tokenizer.encode(marker + "\n"))
            if current_cost + marker_cost <= budget:
                final_parts.insert(0, TextPart(text=marker))

        return final_parts

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
