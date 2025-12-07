"""
Event rendering utilities for LLM consumption.

Converts agex events into provider message formats for LLM communication.
"""

from datetime import datetime
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
    LOW_DETAIL_BUDGET,
    render_action_markdown,
    render_fail,
    render_output_parts_full,
    render_success,
    render_summary,
    render_task_start,
)


def render_events_as_markdown(events: List[Event]) -> List[dict]:
    """
    Render events in markdown format (current agex format).

    Returns list of dicts suitable for provider APIs:
        [{"role": "user", "content": "..."}, ...]

    This is the default rendering strategy. Individual clients can use
    this or implement their own rendering (e.g., XML for streaming).

    Automatically applies low-detail rendering to old events based on
    the most recent SummaryEvent's low_detail_threshold.

    Args:
        events: List of Event objects to render

    Returns:
        List of message dicts with role and content
    """
    messages: List[dict[str, Any]] = []

    # Filter out ErrorEvents (not shown to agents)
    filtered_events = [e for e in events if not isinstance(e, ErrorEvent)]

    # Find low-detail threshold from most recent SummaryEvent
    low_detail_threshold: datetime | None = None
    for event in reversed(filtered_events):
        if isinstance(event, SummaryEvent) and event.low_detail_threshold:
            low_detail_threshold = event.low_detail_threshold
            break

    for event in filtered_events:
        # Determine if this event should use low-detail rendering
        use_low_detail = (
            low_detail_threshold is not None
            and event.timestamp < low_detail_threshold
            and isinstance(event, (TaskStartEvent, OutputEvent, SuccessEvent))
        )
        budget = LOW_DETAIL_BUDGET if use_low_detail else HI_DETAIL_BUDGET

        if isinstance(event, TaskStartEvent):
            # Render task start message with appropriate budget
            text, _ = render_task_start(event.message, budget=budget)
            messages.append({"role": "user", "content": text})

        elif isinstance(event, ActionEvent):
            # ActionEvent always renders at full detail (code is compact already)
            text, _ = render_action_markdown(event.thinking, event.code, event.title)
            messages.append({"role": "assistant", "content": text})

        elif isinstance(event, OutputEvent):
            # Render OutputEvent parts with budget (low detail replaces images with placeholders)
            content_parts, _ = render_output_parts_full(event.parts, budget=budget)

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
            # Render success marker with appropriate budget
            text, _ = render_success(event.result, budget=budget)
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
