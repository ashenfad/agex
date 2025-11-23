"""
Event rendering utilities for LLM consumption.

Converts agex events into provider message formats for LLM communication.
"""

from typing import Any, List

from agex.agent.events import (
    ActionEvent,
    ErrorEvent,
    Event,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.llm.core import ContentPart, ImagePart, TextPart
from agex.render.context import ContextRenderer


def render_events_as_markdown(
    events: List[Event],
    model_name: str,
    max_tokens: int,
) -> List[dict]:
    """
    Render events in markdown format (current agex format).

    Returns list of dicts suitable for provider APIs:
        [{"role": "user", "content": "..."}, ...]

    This is the default rendering strategy. Individual clients can use
    this or implement their own rendering (e.g., XML for streaming).

    Args:
        events: List of Event objects to render
        model_name: Model name for tokenization
        max_tokens: Token budget for output rendering

    Returns:
        List of message dicts with role and content
    """
    # ContextRenderer is initialized with model_name for tokenization.
    # Token budget (max_tokens) is passed per render_events() call since
    # different OutputEvents may have different budgets depending on context.
    context_renderer = ContextRenderer(model_name)
    messages: List[dict[str, Any]] = []

    # Filter out ErrorEvents (not shown to agents)
    filtered_events = [e for e in events if not isinstance(e, ErrorEvent)]

    for event in filtered_events:
        if isinstance(event, TaskStartEvent):
            messages.append({"role": "user", "content": event.message})

        elif isinstance(event, ActionEvent):
            # Current markdown format
            content = (
                f"# Thinking\n{event.thinking}\n\n"
                f"# Code\n```python\n{event.code}\n```"
            )
            messages.append({"role": "assistant", "content": content})

        elif isinstance(event, OutputEvent):
            # ContextRenderer handles OutputEvent.parts (list[Any])
            # Returns list[ContentPart] (TextPart or ImagePart)
            content_parts = context_renderer.render_events([event], max_tokens)

            if content_parts:
                # Check for images
                has_images = any(isinstance(p, ImagePart) for p in content_parts)

                if has_images:
                    # Multimodal message - return structured content
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                _content_part_to_dict(part) for part in content_parts
                            ],
                        }
                    )
                else:
                    # Text-only message (all parts are TextPart since has_images is False)
                    text = "\n".join(p.text for p in content_parts)
                    messages.append({"role": "user", "content": text})

        elif isinstance(event, SuccessEvent):
            # Render success marker - use same settings as OutputEvent rendering
            # Convert token budget to character estimate (roughly 4 chars per token)
            from agex.render.value import ValueRenderer

            # Use same max_len and max_depth as StreamRenderer uses for OutputEvent
            estimated_chars = (
                max_tokens * 4
            )  # Conservative estimate: ~4 chars per token
            renderer = ValueRenderer(max_len=estimated_chars, max_depth=4)
            rendered = renderer.render(event.result)
            messages.append(
                {"role": "assistant", "content": f"✅ Task completed: {rendered}"}
            )

        elif isinstance(event, FailEvent):
            messages.append(
                {"role": "assistant", "content": f"❌ Task failed: {event.message}"}
            )

    return messages


def _content_part_to_dict(part: ContentPart) -> dict:
    """
    Convert ContentPart to a generic dict format.

    Individual clients will need to convert this to their provider's format.
    """
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    elif isinstance(part, ImagePart):
        return {"type": "image", "image_data": part.image}
    else:
        raise ValueError(f"Unknown content part type: {type(part)}")
