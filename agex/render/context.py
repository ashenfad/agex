from typing import Any

from agex.agent.events import ErrorEvent, OutputEvent

from ..llm.core import ContentPart, TextPart
from .stream import StreamRenderer


class ContextRenderer:
    """
    Renders the current agent context (recent state changes and print output)
    into a list of multimodal ContentParts suitable for an LLM prompt,
    respecting a token budget.
    """

    def __init__(self, model_name: str):
        self._stream_renderer = StreamRenderer(model_name)

    def render_events(self, events: list[Any], budget: int) -> list[ContentPart]:
        """
        Renders a list of events into a list of ContentParts.
        This is used to create the user message from the agent's recent outputs.
        """
        all_parts: list[ContentPart] = []
        items_to_render = []

        for event in events:
            if isinstance(event, OutputEvent):
                items_to_render.extend(event.parts)
            elif isinstance(event, ErrorEvent):
                # Wrap the error message in a TextPart-like object for rendering
                items_to_render.append(TextPart(text=str(event.error)))

        if not items_to_render:
            return []

        # Render the collected items using the stream renderer
        header = "Agent stdout:\n"
        header_cost = self._stream_renderer.tokenizer.encode(header)
        item_budget = budget - len(header_cost)

        if item_budget > 0:
            rendered_parts = self._stream_renderer.render_item_stream(
                items=items_to_render,
                budget=item_budget,
            )
            if rendered_parts:
                all_parts.append(TextPart(text=header))
                all_parts.extend(rendered_parts)

        return all_parts
