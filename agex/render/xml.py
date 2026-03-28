"""
XML rendering utilities for events.

Converts agex events into XML-formatted messages for LLM consumption.
"""

from typing import Any, List

from agex.agent.datatypes import EditAction, FileAction
from agex.agent.events import (
    ActionEvent,
    CancelledEvent,
    ChapterEvent,
    ClarifyEvent,
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
from agex.llm.xml import (
    TAG_CANCELLED,
    TAG_CLARIFY,
    TAG_EDIT,
    TAG_FAIL,
    TAG_FILE,
    TAG_INSERT_AFTER,
    TAG_INSERT_BEFORE,
    TAG_OBSERVATION,
    TAG_PYTHON,
    TAG_REPLACE,
    TAG_SEARCH,
    TAG_SUCCESS,
    TAG_TERMINAL,
    TAG_THINKING,
    TAG_TITLE,
)
from agex.render.primitives import (
    HI_DETAIL_BUDGET,
    collapse_same_role_messages,
    render_chapter,
    render_output_parts_full,
    render_task_start,
)
from agex.render.value import render_value


def render_events_as_xml(events: List[Event]) -> List[dict]:
    """
    Render events in XML format for LLM consumption.

    Similar to render_events_as_markdown() but uses XML tags.

    Args:
        events: List of Event objects to render

    Returns:
        List of message dicts with role and content
    """
    messages: List[dict[str, Any]] = []

    # Filter out ErrorEvents (not shown to agents)
    filtered_events = [e for e in events if not isinstance(e, ErrorEvent)]

    task_number = 0
    for event in filtered_events:
        budget = HI_DETAIL_BUDGET

        if isinstance(event, TaskStartEvent):
            task_number += 1
            prefix = f"[{task_number}] "
        else:
            prefix = ""

        if isinstance(event, TaskStartEvent):
            # Render task start message with appropriate budget
            text, _ = render_task_start(event.message, budget=budget)
            messages.append({"role": "user", "content": prefix + text})

        elif isinstance(event, ActionEvent):
            # ActionEvent always renders at full detail (XML format with uppercase tags)
            title_section = (
                f"<{TAG_TITLE}>{event.title}</{TAG_TITLE}>" if event.title else ""
            )

            # Render file actions (both FILE and EDIT)
            files_section = ""
            if event.file_actions:
                for action in event.file_actions:
                    if isinstance(action, EditAction):
                        match_all_attr = ' match_all="true"' if action.match_all else ""
                        # Determine which tag to use based on operation
                        if action.operation == "insert-after":
                            content_tag = TAG_INSERT_AFTER
                        elif action.operation == "insert-before":
                            content_tag = TAG_INSERT_BEFORE
                        else:  # "replace"
                            content_tag = TAG_REPLACE
                        files_section += (
                            f'<{TAG_EDIT} path="{action.path}"{match_all_attr}>\n'
                            f"<{TAG_SEARCH}>{action.search}</{TAG_SEARCH}>\n"
                            f"<{content_tag}>{action.content}</{content_tag}>\n"
                            f"</{TAG_EDIT}>\n"
                        )
                    elif isinstance(action, FileAction):
                        mode_attr = (
                            f' mode="{action.mode}"' if action.mode != "write" else ""
                        )
                        files_section += (
                            f'<{TAG_FILE} path="{action.path}"{mode_attr}>'
                            f"{action.content}</{TAG_FILE}>\n"
                        )

            # Choose terminal or python based on what's present
            if event.terminal:
                action_section = f"<{TAG_TERMINAL}>{event.terminal}</{TAG_TERMINAL}>"
            else:
                action_section = f"<{TAG_PYTHON}>{event.code or ''}</{TAG_PYTHON}>"

            content = (
                f"{prefix}{title_section}<{TAG_THINKING}>{event.thinking}</{TAG_THINKING}>\n"
                f"{files_section}"
                f"{action_section}"
            )
            messages.append({"role": "assistant", "content": content})

        elif isinstance(event, OutputEvent):
            # Render OutputEvent parts with budget (low detail replaces images with placeholders)
            content_parts, _ = render_output_parts_full(event.parts, budget=budget)

            if content_parts:
                # Check for images
                has_images = any(isinstance(p, ImagePart) for p in content_parts)

                if has_images:
                    # Multimodal message - wrap in OBSERVATION tags
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{prefix}<{TAG_OBSERVATION}>",
                                },
                                *[
                                    _content_part_to_dict(part)
                                    for part in content_parts
                                ],
                                {"type": "text", "text": f"</{TAG_OBSERVATION}>"},
                            ],
                        }
                    )
                else:
                    # Text-only message
                    text = "\n".join(
                        p.text for p in content_parts if isinstance(p, TextPart)
                    )
                    content = f"{prefix}<{TAG_OBSERVATION}>{text}</{TAG_OBSERVATION}>"
                    messages.append({"role": "user", "content": content})

        elif isinstance(event, SuccessEvent):
            # Render result and wrap in TASK_SUCCESS tag
            estimated_chars = budget * 4
            rendered = render_value(event.result, budget=estimated_chars)

            content = f"{prefix}<{TAG_SUCCESS}>{rendered}</{TAG_SUCCESS}>"
            messages.append({"role": "assistant", "content": content})

        elif isinstance(event, FailEvent):
            content = f"{prefix}<{TAG_FAIL}>{event.message}</{TAG_FAIL}>"
            messages.append({"role": "assistant", "content": content})

        elif isinstance(event, ClarifyEvent):
            content = f"{prefix}<{TAG_CLARIFY}>{event.message}</{TAG_CLARIFY}>"
            messages.append({"role": "assistant", "content": content})

        elif isinstance(event, CancelledEvent):
            content = f"{prefix}<{TAG_CANCELLED}>Task '{event.task_name}' cancelled after {event.iterations_completed} iterations</{TAG_CANCELLED}>"
            messages.append({"role": "user", "content": content})

        elif isinstance(event, ChapterEvent):
            text, _ = render_chapter(event.name, event.message)
            messages.append({"role": "user", "content": prefix + text})

        elif isinstance(event, SystemNoteEvent):
            # Render system note as a user message (transient context)
            messages.append({"role": "user", "content": prefix + event.message})

        elif isinstance(event, FileEvent):
            # Render file changes with XML tags
            parts = []
            if event.added:
                parts.append(f"Added: {', '.join(event.added)}")
            if event.modified:
                parts.append(f"Modified: {', '.join(event.modified)}")
            if event.removed:
                parts.append(f"Removed: {', '.join(event.removed)}")
            content = f"{prefix}<FILE_CHANGES source='{event.file_source}'>{'; '.join(parts)}</FILE_CHANGES>"
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
