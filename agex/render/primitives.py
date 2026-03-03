"""
Rendering primitives for events and values.

This module provides low-level rendering functions that don't depend on event types,
enabling clean layering: primitives → events → provider messages.
"""

import base64
import io
from typing import Any

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

from ..agent.datatypes import EditAction, FileAction
from ..eval.objects import ImageAction, PrintAction
from ..fs.slugify import slugify as _slugify
from ..llm.core import ContentPart, ImagePart, TextPart
from ..tokenizers import get_tokenizer
from .value import render_value

# Standard token budget for "hi" detail rendering
HI_DETAIL_BUDGET = 8192

# Low-detail budget for older events (roughly 1/4 of high-detail)
LOW_DETAIL_BUDGET = 1024


# ============================================================================
# Shared DataFrame utilities
# ============================================================================


def is_dataframe(value: Any) -> bool:
    """
    Check if a value is a pandas DataFrame.

    Uses duck-typing to avoid hard pandas dependency.
    Excludes list/dict/set/tuple which might have shape/columns attributes.

    Args:
        value: Value to check

    Returns:
        True if value appears to be a pandas DataFrame
    """
    return (
        hasattr(value, "shape")
        and hasattr(value, "columns")
        and not isinstance(value, (list, dict, set, tuple))
    )


def render_dataframe_with_budget(value: Any, token_budget: int | None) -> str:
    """
    Render a DataFrame to string with optimal pandas display settings.

    Uses iterative token counting to find the best row limit within budget.
    This is a shared utility used by render_value.

    Args:
        value: The DataFrame to render
        token_budget: Optional token budget for optimization. If None, uses
                     conservative column-based defaults.

    Returns:
        String representation of the DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        return str(value)

    old_max_rows = pd.options.display.max_rows
    old_min_rows = pd.options.display.min_rows

    try:
        if token_budget is not None:
            # Iterative token-counting approach
            tokenizer = get_tokenizer("gpt-4")
            num_rows = getattr(value, "shape")[0]

            # Generate smart candidates based on DataFrame size
            # Always include actual row count as first candidate
            if num_rows <= 100:
                candidates = [num_rows, 80, 60, 40, 20]
            elif num_rows <= 200:
                candidates = [num_rows, 200, 150, 100, 60, 40]
            else:
                candidates = [200, 150, 120, 80, 60, 40]

            # Try candidates from largest to smallest
            best_limit = 40
            for limit in candidates:
                if limit > num_rows:
                    continue

                pd.options.display.max_rows = limit
                pd.options.display.min_rows = limit
                test_str = str(value)
                test_tokens = len(tokenizer.encode(test_str))

                if test_tokens <= token_budget:
                    best_limit = limit
                    break

            pd.options.display.max_rows = best_limit
            pd.options.display.min_rows = best_limit
        else:
            # No budget: use conservative defaults based on column count
            num_cols = len(getattr(value, "columns"))
            limit = 200 if num_cols <= 5 else 120 if num_cols <= 10 else 60
            pd.options.display.max_rows = limit
            pd.options.display.min_rows = limit

        return str(value)
    finally:
        pd.options.display.max_rows = old_max_rows
        pd.options.display.min_rows = old_min_rows


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken with gpt-4 encoding.

    This provides a model-agnostic token estimate suitable for budgeting.
    """
    tokenizer = get_tokenizer("gpt-4")
    return len(tokenizer.encode(text))


# ============================================================================
# Image utilities
# ============================================================================


def _is_plotly_figure(image: Any) -> bool:
    """Check if an object is a Plotly figure using duck typing."""
    # Check for to_image method (defining characteristic of Plotly figures)
    if hasattr(image, "to_image") and callable(getattr(image, "to_image", None)):
        # Also check for layout attribute (Plotly figures have this)
        if hasattr(image, "layout"):
            return True
    # Fallback: check isinstance if plotly is available
    if plotly is not None:
        try:
            return isinstance(image, plotly.graph_objects.Figure)
        except Exception:
            pass
    return False


def estimate_image_cost(image: Any, detail: str = "high") -> int:
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
    elif _is_plotly_figure(image):
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


def serialize_image_to_base64(image: Any) -> str | None:
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

        if _is_plotly_figure(image):
            # kaleido is used by plotly to export static images
            # Use duck typing - check for to_image method
            if hasattr(image, "to_image") and callable(
                getattr(image, "to_image", None)
            ):
                image_bytes = image.to_image(format="png")
                return base64.b64encode(image_bytes).decode("utf-8")
    except Exception:
        # If any error occurs during serialization, fail gracefully.
        # The caller will generate appropriate error messages
        return None

    # Unsupported type
    return None


def get_image_error_message(image: Any) -> str:
    """Generate a helpful error message for failed image serialization."""
    if not _is_plotly_figure(image):
        return f"<unsupported image type: {type(image).__name__}>"

    # Try to get the actual error from Plotly export
    error_msg = None
    try:
        if hasattr(image, "to_image") and callable(getattr(image, "to_image", None)):
            image.to_image(format="png")
    except Exception as e:
        error_msg = str(e)

    # Check for kaleido-specific errors
    if error_msg and ("kaleido" in error_msg.lower()):
        return (
            "<Plotly figure export failed: Kaleido package is required. "
            "Install with: pip install kaleido>"
        )
    elif error_msg:
        return f"<Plotly figure export failed: {error_msg}>"
    else:
        return (
            "<Plotly figure export failed: Kaleido package may be missing. "
            "Install with: pip install kaleido>"
        )


# ============================================================================
# Event rendering functions
# ============================================================================


def render_action_markdown(
    thinking: str,
    code: str | None = None,
    title: str = "",
    file_actions: list[FileAction | EditAction] | None = None,
    terminal: str | None = None,
) -> tuple[str, int]:
    """
    Render an action event as markdown.

    Args:
        thinking: The agent's thinking/reasoning
        code: Python code to execute (None if terminal used)
        title: Optional title for the action
        file_actions: List of file write/edit actions
        terminal: Terminal script to execute (mutually exclusive with code)

    Returns:
        (markdown_text, token_count)
    """
    title_section = f"# {title}\n" if title else ""
    files_section = ""

    if file_actions:
        files_section = "## Files\n"
        for action in file_actions:
            if isinstance(action, EditAction):
                match_all_str = " (match_all)" if action.match_all else ""
                op_str = (
                    f" ({action.operation})" if action.operation != "replace" else ""
                )
                files_section += f"### EDIT {action.path}{op_str}{match_all_str}\n"
                files_section += f"Search:\n```\n{action.search}\n```\n"
                # Label based on operation
                if action.operation == "insert-after":
                    label = "Insert After"
                elif action.operation == "insert-before":
                    label = "Insert Before"
                else:
                    label = "Replace"
                files_section += f"{label}:\n```\n{action.content}\n```\n\n"
            else:
                path, content, mode = action.path, action.content, action.mode
                mode_suffix = f" (mode: {mode})" if mode != "write" else ""
                files_section += f"### {path}{mode_suffix}\n{content}\n\n"

    # Render terminal or code section
    if terminal:
        action_section = f"# Terminal\n```bash\n{terminal}\n```"
    else:
        action_section = f"# Code\n```python\n{code or ''}\n```"

    content = (
        f"{title_section}# Thinking\n{thinking}\n\n{files_section}{action_section}"
    )
    tokens = count_tokens(content)
    return content, tokens


def render_task_start(message: str, budget: int = HI_DETAIL_BUDGET) -> tuple[str, int]:
    """
    Render a task start event.

    Args:
        message: The task start message (may contain rich input rendering)
        budget: Token budget for rendering (HI_DETAIL_BUDGET or LOW_DETAIL_BUDGET)

    Returns:
        (message_text, token_count)

    Note: Currently message is pre-rendered, so budget doesn't affect it.
    Future enhancement could re-render inputs at different detail levels.
    """
    tokens = count_tokens(message)
    return message, tokens


def render_success(result: Any, budget: int = HI_DETAIL_BUDGET) -> tuple[str, int]:
    """
    Render a success event with result value.

    Args:
        result: The task result to render
        budget: Token budget for rendering (HI_DETAIL_BUDGET or LOW_DETAIL_BUDGET)

    Returns:
        (success_text, token_count)
    """
    estimated_chars = budget * 4  # ~4 chars per token
    rendered = render_value(result, budget=estimated_chars)
    text = f"✅ Task completed: {rendered}"
    tokens = count_tokens(text)
    return text, tokens


def render_fail(message: str) -> tuple[str, int]:
    """
    Render a fail event.

    Returns:
        (fail_text, token_count)
    """
    text = f"❌ Task failed: {message}"
    tokens = count_tokens(text)
    return text, tokens


def render_chapter(name: str, message: str) -> tuple[str, int]:
    """
    Render a chapter event.

    Returns:
        (chapter_text, token_count)
    """
    path = f"/chapters/{_slugify(name)}/"

    text = f'📖 Chapter: "{name}"\n\n{message}\n\nFull details: {path}'
    tokens = count_tokens(text)
    return text, tokens


def render_output_parts_full(
    parts: list[Any], budget: int = HI_DETAIL_BUDGET
) -> tuple[list[ContentPart], int]:
    """
    Render OutputEvent parts with budget management.

    This renders PrintActions, ImageActions, and other objects with a configurable budget,
    returning the actual token count of what was rendered.

    Args:
        parts: List of PrintAction, ImageAction, or other objects
        budget: Token budget for rendering (HI_DETAIL_BUDGET or LOW_DETAIL_BUDGET)

    Returns:
        (content_parts, actual_token_count)
    """
    if not parts:
        return [], 0

    tokenizer = get_tokenizer("gpt-4")

    char_budget = budget * 4  # ~4 chars per token

    def render_func(v):
        return render_value(v, budget=char_budget, token_budget=budget)

    # Store tuples of (ContentPart, cost) to manage budget.
    parts_with_cost: list[tuple[ContentPart, int]] = []
    current_cost = 0
    omitted_items = False

    for item in reversed(parts):
        part: ContentPart | None = None
        cost = 0

        if isinstance(item, PrintAction):
            rendered_args = [render_func(arg) for arg in item]
            rendered_line = " ".join(map(str, rendered_args))
            cost = len(tokenizer.encode(rendered_line + "\n"))
            part = TextPart(text=rendered_line)

        elif isinstance(item, ImageAction):
            # Low detail: replace images with text placeholders
            if budget == LOW_DETAIL_BUDGET:
                placeholder = "[Image]"
                cost = len(tokenizer.encode(placeholder + "\n"))
                part = TextPart(text=placeholder)
            else:
                # High detail: include the actual image
                cost = estimate_image_cost(item.image, item.detail)
                # Only serialize if it might fit.
                if current_cost + cost <= budget:
                    base64_image = serialize_image_to_base64(item.image)
                    if base64_image:
                        part = ImagePart(image=base64_image)
                    else:
                        # Generate error message based on image type
                        placeholder = get_image_error_message(item.image)
                        cost = len(tokenizer.encode(placeholder + "\n"))
                        part = TextPart(text=placeholder)
                else:
                    cost = 0  # Reset cost, we are not adding this part

        else:  # Fallback for other raw types in the stream
            rendered_line = render_func(item)
            cost = len(tokenizer.encode(rendered_line + "\n"))
            part = TextPart(text=rendered_line)

        if part and current_cost + cost <= budget:
            parts_with_cost.insert(0, (part, cost))
            current_cost += cost
        elif part and isinstance(part, TextPart) and cost > 0:
            # Text part exceeds remaining budget - truncate to fit
            # Reserve space for truncation marker
            marker = "... [output truncated]"
            marker_cost = len(tokenizer.encode(marker + "\n"))
            available_tokens = budget - current_cost - marker_cost

            if (
                available_tokens > 100
            ):  # Only truncate if we can show something meaningful
                # Binary search to find the right truncation point
                text = part.text
                low, high = 0, len(text)
                best_text = ""
                while low < high:
                    mid = (low + high + 1) // 2
                    candidate = text[:mid]
                    candidate_tokens = len(tokenizer.encode(candidate + "\n"))
                    if candidate_tokens <= available_tokens:
                        best_text = candidate
                        low = mid
                    else:
                        high = mid - 1

                if best_text:
                    truncated_text = best_text + "\n" + marker
                    truncated_cost = len(tokenizer.encode(truncated_text + "\n"))
                    parts_with_cost.insert(
                        0, (TextPart(text=truncated_text), truncated_cost)
                    )
                    current_cost += truncated_cost
                    omitted_items = True
                else:
                    omitted_items = True
            else:
                omitted_items = True
        elif cost > 0:  # If we calculated a cost but didn't add the part
            omitted_items = True

    # Post-processing: add truncation markers for completely omitted parts
    final_parts: list[ContentPart] = [p for p, c in parts_with_cost]

    if omitted_items and not final_parts:
        # Nothing was added but there was content - add a message
        placeholder = "[Output exceeded display budget and was truncated]"
        placeholder_cost = len(tokenizer.encode(placeholder + "\n"))
        if placeholder_cost <= budget:
            final_parts.append(TextPart(text=placeholder))
            current_cost = placeholder_cost
    elif omitted_items and final_parts:
        marker = "..."
        marker_cost = len(tokenizer.encode(marker + "\n"))
        if current_cost + marker_cost <= budget:
            final_parts.insert(0, TextPart(text=marker))
            current_cost += marker_cost

    return final_parts, current_cost
