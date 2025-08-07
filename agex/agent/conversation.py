"""
Conversation log reconstruction from event streams.

This module provides functionality to reconstruct conversation logs from the unified
event system, with filtering for different consumer types (agents vs monitoring).
"""

from typing import Sequence

from agex.agent.base import BaseAgent
from agex.agent.events import (
    ActionEvent,
    ErrorEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.llm.core import Message, TextMessage
from agex.render.context import ContextRenderer
from agex.state.core import State


def conversation_log(
    state: State, system_message: str, agent: BaseAgent
) -> Sequence[Message]:
    """
    Reconstruct the full conversation from the event log in state.
    This function renders all events into messages in chronological order.
    """
    from agex.state.log import get_events_from_log

    all_events = get_events_from_log(state)

    # Filter to only agent-relevant events (exclude ErrorEvents)
    event_log = [event for event in all_events if not isinstance(event, ErrorEvent)]

    messages: list[Message] = [TextMessage(role="system", content=system_message)]

    # Render all events in chronological order
    context_renderer = ContextRenderer(agent.llm_client.model)

    for event in event_log:
        if isinstance(event, TaskStartEvent):
            # Task start → user message
            messages.append(TextMessage(role="user", content=event.message))

        elif isinstance(event, ActionEvent):
            # Agent action → assistant message
            assistant_content = (
                f"# Thinking\n{event.thinking}\n\n# Code\n```python\n{event.code}\n```"
            )
            messages.append(TextMessage(role="assistant", content=assistant_content))

        elif isinstance(event, OutputEvent):
            # Agent output → user message (rendered by context renderer)
            context_parts = context_renderer.render_events([event], agent.max_tokens)
            if context_parts:
                # Check if there are any non-text parts (e.g., ImageParts)
                from agex.llm.core import ImagePart, MultimodalMessage, TextPart

                has_non_text_parts = any(
                    isinstance(part, ImagePart) for part in context_parts
                )

                if has_non_text_parts:
                    # Create a MultimodalMessage with all parts
                    messages.append(
                        MultimodalMessage(role="user", content=context_parts)
                    )
                else:
                    # All parts are text, create a TextMessage
                    full_text = "\n".join(
                        part.text
                        for part in context_parts
                        if isinstance(part, TextPart)
                    )
                    messages.append(TextMessage(role="user", content=full_text))

        elif isinstance(event, SuccessEvent):
            # Agent success → assistant message (with safe rendering)
            from agex.render.value import ValueRenderer

            renderer = ValueRenderer(max_len=200, max_depth=2)
            rendered_result = renderer.render(event.result)
            assistant_content = f"✅ Task completed successfully: {rendered_result}"
            messages.append(TextMessage(role="assistant", content=assistant_content))

        elif isinstance(event, FailEvent):
            # Agent failure → assistant message
            assistant_content = f"❌ Task failed: {event.message}"
            messages.append(TextMessage(role="assistant", content=assistant_content))

    return messages
