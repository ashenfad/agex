import base64
import io
from typing import Any, List, Optional

# Gracefully import optional image libraries
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import matplotlib.figure
except ImportError:
    matplotlib = None  # type: ignore

try:
    import plotly.graph_objects
except ImportError:
    plotly = None  # type: ignore


from ..eval.objects import ImageAction, PrintAction
from ..llm.core import ContentPart, ImagePart, TextPart
from ..tokenizers import Tokenizer, get_tokenizer
from .value import ValueRenderer


def _estimate_image_cost(image: Any, detail: str = "high") -> int:
    """
    Estimates the token cost for an image.

    This provides a reasonable, model-agnostic estimation for budget management.

    Args:
        image: The image object (e.g., PIL Image, Matplotlib Figure).
        detail: The requested detail level ("high" or "low").

    Returns:
        The estimated token cost.
    """
    if detail == "low":
        return 85  # A common, fixed cost for low-detail/thumbnail images.

    # For high detail, we need the image dimensions.
    width, height = 0, 0
    if Image and isinstance(image, Image.Image):
        width, height = image.size
    elif matplotlib and isinstance(image, matplotlib.figure.Figure):
        # Matplotlib figures are in inches; convert to pixels using a common default DPI.
        dpi = image.get_dpi() if image.get_dpi() else 100.0
        width, height = (
            int(image.get_figwidth() * dpi),
            int(image.get_figheight() * dpi),
        )
    elif plotly and isinstance(image, plotly.graph_objects.Figure):
        # Plotly figures often have explicit pixel dimensions.
        width = image.layout.width if image.layout.width else 500
        height = image.layout.height if image.layout.height else 400
    else:
        # Fallback for unsupported types: a fixed high-cost guess.
        return 2000

    if width == 0 or height == 0:
        return 2000  # Avoid division by zero for invalid images

    # Use a simple, linear scaling formula as a general-purpose heuristic.
    # Anthropic's is (width_px * height_px) / 750, which is a good baseline.
    return (width * height) // 750


def _serialize_image_to_base64(image: Any) -> Optional[str]:
    """Serializes a supported image type to a PNG base64 string."""
    buffer = io.BytesIO()
    try:
        if Image and isinstance(image, Image.Image):
            # For security and consistency, convert to a standard format like PNG.
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        elif matplotlib and isinstance(image, matplotlib.figure.Figure):
            image.savefig(buffer, format="png", bbox_inches="tight")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        elif plotly and isinstance(image, plotly.graph_objects.Figure):
            # kaleido is used by plotly to export static images
            image_bytes = image.to_image(format="png")
            return base64.b64encode(image_bytes).decode("utf-8")
    except Exception:
        # If any error occurs during serialization, fail gracefully.
        return None

    # Unsupported type
    return None


class StreamRenderer:
    """
    Renders streams of Python objects into strings or multimodal content parts,
    respecting a token budget. This class is responsible for low-level rendering.
    """

    def __init__(self, model_name: str):
        self.tokenizer: Tokenizer = get_tokenizer(model_name)
        self.value_renderer = ValueRenderer(max_len=4096)

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
                cost = _estimate_image_cost(item.image, item.detail)
                # Only serialize if it might fit.
                if current_cost + cost <= budget:
                    base64_image = _serialize_image_to_base64(item.image)
                    if base64_image:
                        part = ImagePart(image=base64_image)
                    else:
                        # Fallback to a text placeholder if serialization fails.
                        placeholder = (
                            f"<unsupported image type: {type(item.image).__name__}>"
                        )
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
