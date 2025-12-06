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
    SummaryEvent,
    TaskStartEvent,
)
from agex.llm.core import ContentPart, ImagePart, TextPart
from agex.render.primitives import (
    HI_DETAIL_BUDGET,
    render_fail,
    render_output_parts_full,
    render_success,
    render_summary,
)


def render_events_as_markdown(events: List[Event]) -> List[dict]:
    """
    Render events in markdown format (current agex format).

    Returns list of dicts suitable for provider APIs:
        [{"role": "user", "content": "..."}, ...]

    This is the default rendering strategy. Individual clients can use
    this or implement their own rendering (e.g., XML for streaming).

    Args:
        events: List of Event objects to render

    Returns:
        List of message dicts with role and content
    """
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
            # Render OutputEvent parts at hi-detail level
            content_parts, _ = render_output_parts_full(event.parts, HI_DETAIL_BUDGET)

            if content_parts:
                # Add "Agent stdout:" header
                header = TextPart(text="Agent stdout:")
                all_parts = [header] + content_parts

                # Check for images
                has_images = any(isinstance(p, ImagePart) for p in all_parts)

                if has_images:
                    # Multimodal message - return structured content
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                _content_part_to_dict(part) for part in all_parts
                            ],
                        }
                    )
                else:
                    # Text-only message (all parts are TextPart since has_images is False)
                    text = "\n".join(
                        p.text for p in all_parts if isinstance(p, TextPart)
                    )
                    messages.append({"role": "user", "content": text})

        elif isinstance(event, SuccessEvent):
            # Render success marker using primitives
            text, _ = render_success(event.result)
            messages.append({"role": "assistant", "content": text})

        elif isinstance(event, FailEvent):
            # Render fail marker using primitives
            text, _ = render_fail(event.message)
            messages.append({"role": "assistant", "content": text})

        elif isinstance(event, SummaryEvent):
            # Render summary event using primitives
            text, _ = render_summary(
                event.summary, event.summarized_event_count, event.original_tokens
            )
            messages.append({"role": "user", "content": text})

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
