"""
Event rendering utilities for LLM consumption.

Converts agex events into provider message formats for LLM communication.
"""

from typing import Any, List

from agex.agent.events import (
    ActionEvent,
    ChapterEvent,
    ErrorEvent,
    Event,
    FailEvent,
    FileEvent,
    OutputEvent,
    SuccessEvent,
    SystemNoteEvent,
    TaskStartEvent,
)
from agex.llm.core import ContentPart, ImagePart, TextPart
from agex.render.primitives import (
    HI_DETAIL_BUDGET,
    collapse_same_role_messages,
    render_action_markdown,
    render_chapter,
    render_output_parts_full,
    render_task_start,
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

    for i, event in enumerate(filtered_events, 1):
        budget = HI_DETAIL_BUDGET
        prefix = f"[{i}] "

        if isinstance(event, TaskStartEvent):
            text, _ = render_task_start(event.message, budget=budget)
            messages.append({"role": "user", "content": prefix + text})

        elif isinstance(event, ActionEvent):
            # ActionEvent always renders at full detail (code is compact already)
            text, _ = render_action_markdown(
                event.thinking,
                event.code,
                event.title,
                event.file_actions,
                event.terminal,
            )
            messages.append({"role": "assistant", "content": prefix + text})

        elif isinstance(event, OutputEvent):
            # Render OutputEvent parts with budget (low detail replaces images with placeholders)
            content_parts, _ = render_output_parts_full(event.parts, budget=budget)

            if content_parts:
                # Add numbered "Agent stdout:" header
                header = TextPart(text=prefix + "Agent stdout:")
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

        elif isinstance(event, (SuccessEvent, FailEvent)):
            # Terminal events are not rendered — the LLM already expressed
            # its intent via task_success/task_fail in the preceding
            # ActionEvent's code.
            pass

        elif isinstance(event, ChapterEvent):
            text, _ = render_chapter(event.name, event.message)
            messages.append({"role": "assistant", "content": prefix + text})

        elif isinstance(event, SystemNoteEvent):
            # Render system note as a user message (transient context)
            messages.append({"role": "user", "content": prefix + event.message})

        elif isinstance(event, FileEvent):
            # Render file changes compactly
            parts = []
            if event.added:
                parts.append(f"Added: {', '.join(event.added)}")
            if event.modified:
                parts.append(f"Modified: {', '.join(event.modified)}")
            if event.removed:
                parts.append(f"Removed: {', '.join(event.removed)}")
            content = (
                prefix + f"[File changes by {event.file_source}] " + "; ".join(parts)
            )
            messages.append({"role": "user", "content": content})

    return collapse_same_role_messages(messages)


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
