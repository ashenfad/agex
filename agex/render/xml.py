"""
XML rendering utilities for events.

Converts agex events into XML-formatted messages for LLM consumption.
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
from agex.llm.xml import TAG_PYTHON, TAG_THINKING, TAG_TITLE
from agex.render.context import ContextRenderer


def render_events_as_xml(
    events: List[Event],
    model_name: str,
    max_tokens: int,
) -> List[dict]:
    """
    Render events in XML format for LLM consumption.

    Similar to render_events_as_markdown() but uses XML tags.
    Clients can use this or implement their own rendering.

    Args:
        events: List of Event objects to render
        model_name: Model name for tokenization
        max_tokens: Token budget for output rendering

    Returns:
        List of message dicts with role and content
    """
    context_renderer = ContextRenderer(model_name)
    messages: List[dict[str, Any]] = []

    # Filter out ErrorEvents (not shown to agents)
    filtered_events = [e for e in events if not isinstance(e, ErrorEvent)]

    for event in filtered_events:
        if isinstance(event, TaskStartEvent):
            messages.append({"role": "user", "content": event.message})

        elif isinstance(event, ActionEvent):
            # XML format with uppercase tags
            title_section = (
                f"<{TAG_TITLE}>{event.title}</{TAG_TITLE}>" if event.title else ""
            )
            content = (
                f"{title_section}<{TAG_THINKING}>{event.thinking}</{TAG_THINKING}>\n"
                f"<{TAG_PYTHON}>{event.code}</{TAG_PYTHON}>"
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
                    text = "\n".join(
                        p.text for p in content_parts if isinstance(p, TextPart)
                    )
                    messages.append({"role": "user", "content": text})

        elif isinstance(event, SuccessEvent):
            # Render success marker
            from agex.render.value import ValueRenderer

            renderer = ValueRenderer(max_len=200, max_depth=2)
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
