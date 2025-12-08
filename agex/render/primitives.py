"""
Rendering primitives for events and values.

This module provides low-level rendering functions that don't depend on event types,
enabling clean layering: primitives → events → provider messages.
"""

import base64
import io
from dataclasses import fields, is_dataclass
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

from ..eval.objects import AgexClass, AgexInstance, AgexObject, ImageAction, PrintAction
from ..llm.core import ContentPart, ImagePart, TextPart
from ..tokenizers import get_tokenizer

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
    This is a shared utility used by both ValueRenderer classes.

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
# ValueRenderer - renders Python values to strings
# ============================================================================


class ValueRenderer:
    """Renders any Python value into a string suitable for an LLM prompt."""

    def __init__(
        self,
        max_len: int = 2048,
        max_depth: int = 2,
        max_items: int = 50,
        token_budget: int | None = None,
    ):
        self.max_len = max_len
        self.max_depth = max_depth
        self.max_items = max_items
        self.token_budget = token_budget  # Optional token budget for smart rendering

    def render(self, value: Any, current_depth: int = 0, compact: bool = False) -> str:
        """
        Renders a value to a string, dispatching to type-specific helpers.

        Args:
            value: The value to render
            current_depth: Current nesting depth
            compact: If True, use compact representations suitable for inline display
        """
        # Handle custom types first (using duck typing to avoid circular imports)
        if hasattr(value, "__class__") and type(value).__name__ == "UserFunction":
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
        # But skip depth check for display objects like DataFrames (they should show content)
        # DataFrames have shape and columns attributes, making them display objects
        is_display_object = (
            hasattr(value, "shape")
            and hasattr(value, "columns")
            and not isinstance(value, (list, dict, set, tuple))
        )

        if current_depth >= self.max_depth and not is_display_object:
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

    def _render_user_function(self, value: Any) -> str:
        """Render a UserFunction object (duck typed to avoid circular imports)."""
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
            str_repr = self._render_with_display_options(value)
        except Exception:
            return self._render_metadata_fallback(value, type_name)

        # Check if it's actually informative (not default object repr)
        if self._is_default_object_repr(str_repr, type_name):
            return self._render_metadata_fallback(value, type_name)

        # For DataFrames with token budget: skip char-based truncation
        # (already handled by iterative token-counting in _render_with_display_options)
        if is_dataframe(value) and self.token_budget is not None:
            return str_repr  # Already optimally rendered

        # Apply intelligent truncation if needed for other values
        if len(str_repr) <= self.max_len:
            return str_repr

        return self._truncate_intelligently(str_repr)

    def _render_with_display_options(self, value: Any) -> str:
        """
        Render value to string, with special handling for DataFrames.
        """
        if is_dataframe(value):
            return render_dataframe_with_budget(value, self.token_budget)
        else:
            return str(value)

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
            # Use non-compact mode to show full content (e.g., DataFrames)
            rendered_value = self.render(field_value, depth + 1, compact=False)
            item_str = f"{f.name}={rendered_value}"
            if len(str(items)) + len(item_str) > self.max_len:
                items.append("...")
                break
            items.append(item_str)
        return f"{type(value).__name__}({', '.join(items)})"


# ============================================================================
# Event rendering functions
# ============================================================================


def render_action_markdown(
    thinking: str, code: str, title: str = ""
) -> tuple[str, int]:
    """
    Render an action event as markdown.

    Returns:
        (markdown_text, token_count)
    """
    content = f"# Thinking\n{thinking}\n\n# Code\n```python\n{code}\n```"
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
    # Adjust depth based on budget (low detail = shallower depth)
    max_depth = 2 if budget == LOW_DETAIL_BUDGET else 4
    max_items = 10 if budget == LOW_DETAIL_BUDGET else 25

    renderer = ValueRenderer(
        max_len=estimated_chars,
        max_depth=max_depth,
        max_items=max_items,
    )
    rendered = renderer.render(result)
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


def render_summary(
    summary: str, event_count: int, original_tokens: int
) -> tuple[str, int]:
    """
    Render a summary event.

    Returns:
        (summary_text, token_count)
    """
    text = f"📝 Summary of {event_count} previous events (originally {original_tokens} tokens):\n\n{summary}"
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

    # Adjust rendering parameters based on budget
    # Use ~4 chars per token as rough estimate for max_len
    if budget == LOW_DETAIL_BUDGET:
        render_func = ValueRenderer(
            max_len=budget * 4,  # 1024 tokens → 4K chars
            max_depth=2,
            max_items=10,
            token_budget=budget,
        ).render
    else:
        render_func = ValueRenderer(
            max_len=budget * 4,  # 8192 tokens → 32K chars
            max_depth=4,
            token_budget=budget,
        ).render

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
        elif cost > 0:  # If we calculated a cost but didn't add the part
            omitted_items = True

    # Post-processing: add truncation markers
    final_parts: list[ContentPart] = [p for p, c in parts_with_cost]

    if omitted_items and final_parts:
        marker = "..."
        marker_cost = len(tokenizer.encode(marker + "\n"))
        if current_cost + marker_cost <= budget:
            final_parts.insert(0, TextPart(text=marker))
            current_cost += marker_cost

    return final_parts, current_cost
