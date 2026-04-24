"""
Rendering primitives for events and values.

This module provides low-level rendering functions that don't depend on event types,
enabling clean layering: primitives → events → provider messages.

Image-specific helpers (PIL/matplotlib/plotly) live in ``.images`` and are
imported lazily — ``import agex`` reaches this module eagerly via the
agent-event chain, so eager heavy-graphics imports here would pay that
cost on every agex load. Callers that need image utilities import them
from ``.images`` directly.
"""

from typing import Any

from ..agent.emissions import (
    Emission,
    FileEditEmission,
    FileWriteEmission,
    PythonEmission,
    TerminalEmission,
    TextEmission,
    ThinkingEmission,
)
from ..eval.objects import ImageAction, PrintAction
from ..fs.slugify import slugify as _slugify
from ..llm.core import ContentPart, ImagePart, TextPart
from ..tokenizers import get_tokenizer
from .images import (
    estimate_image_cost,
    get_image_error_message,
    serialize_image_to_base64,
)
from .value import render_value

# Re-export for backward compatibility: older code paths and external
# callers import these directly from ``render.primitives``.
__all__ = [
    "HI_DETAIL_BUDGET",
    "LOW_DETAIL_BUDGET",
    "collapse_same_role_messages",
    "count_tokens",
    "estimate_image_cost",
    "get_image_error_message",
    "is_dataframe",
    "render_action_markdown",
    "render_chapter",
    "render_dataframe_with_budget",
    "render_fail",
    "render_output_parts_full",
    "render_success",
    "render_task_start",
    "serialize_image_to_base64",
]

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
# Event rendering functions
# ============================================================================


def render_action_markdown(
    emissions: list[Emission],
) -> tuple[str, int]:
    """Render an action event's emission list as markdown.

    Walks emissions in order and composes a single markdown document
    that is semantically faithful to the old ``render_action_markdown``
    output (title, thinking, report, file sections, code/terminal).
    Used by the summarization path and any caller that wants a
    human-readable transcript of a turn.
    """
    title_parts: list[str] = []
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    file_sections: list[str] = []
    code_sections: list[str] = []
    terminal_sections: list[str] = []

    for em in emissions:
        if isinstance(em, TextEmission):
            if em.text:
                text_parts.append(em.text)
        elif isinstance(em, ThinkingEmission):
            if em.text and not em.redacted:
                thinking_parts.append(em.text)
        elif isinstance(em, PythonEmission):
            if em.title:
                title_parts.append(em.title)
            if em.code:
                code_sections.append(em.code)
        elif isinstance(em, TerminalEmission):
            if em.title:
                title_parts.append(em.title)
            if em.commands:
                terminal_sections.append(em.commands)
        elif isinstance(em, FileEditEmission):
            match_all_str = " (match_all)" if em.match_all else ""
            section = f"### EDIT {em.path}{match_all_str}\n"
            section += f"Search:\n```\n{em.search}\n```\n"
            section += f"Replace:\n```\n{em.content}\n```\n\n"
            file_sections.append(section)
        elif isinstance(em, FileWriteEmission):
            mode_suffix = f" (mode: {em.mode})" if em.mode != "write" else ""
            section = f"### {em.path}{mode_suffix}\n{em.content}\n\n"
            file_sections.append(section)

    title = "; ".join(title_parts)
    thinking = "\n\n".join(thinking_parts)
    report = "\n\n".join(text_parts)

    title_section = f"# {title}\n" if title else ""
    report_section = f"# Report\n{report}\n\n" if report else ""
    files_section = ""
    if file_sections:
        files_section = "## Files\n" + "".join(file_sections)

    if terminal_sections:
        joined = "\n".join(terminal_sections)
        action_section = f"# Terminal\n```bash\n{joined}\n```"
    elif code_sections:
        joined = "\n".join(code_sections)
        action_section = f"# Code\n```python\n{joined}\n```"
    else:
        action_section = ""

    content = (
        f"{title_section}# Thinking\n{thinking}\n\n"
        f"{report_section}{files_section}{action_section}"
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
    rendered = render_value(result, budget=estimated_chars, token_budget=budget)
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


# ============================================================================
# Message collapsing
# ============================================================================


def _merge_content(a, b):
    """Merge two message content values (str or list-of-content-part dicts)."""
    if isinstance(a, str) and isinstance(b, str):
        return a + "\n" + b
    # Normalize to list form then concatenate
    if isinstance(a, str):
        a = [{"type": "text", "text": a}]
    if isinstance(b, str):
        b = [{"type": "text", "text": b}]
    return a + b


def collapse_same_role_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages into single messages."""
    if not messages:
        return []
    collapsed: list[dict] = []
    for msg in messages:
        if collapsed and collapsed[-1]["role"] == msg["role"]:
            collapsed[-1] = {
                "role": msg["role"],
                "content": _merge_content(collapsed[-1]["content"], msg["content"]),
            }
        else:
            collapsed.append(dict(msg))
    return collapsed
