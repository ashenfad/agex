"""
Dummy LLM client for testing purposes.

This module provides a mock LLMClient that returns predefined LLMResponse objects
sequentially, useful for testing agent behavior without actual LLM calls.
"""

from typing import List

from agex.agent.events import Event, OutputEvent

from .core import LLMClient, LLMResponse


class DummyLLMClient(LLMClient):
    """
    A dummy LLM client that returns predefined LLMResponse objects in sequence.
    Useful for testing agent logic without actual LLM calls.
    """

    def __init__(
        self, responses: List[LLMResponse | Exception] | None = None, **kwargs
    ):
        """
        Initialize with a sequence of LLMResponse objects to return.

        Args:
            responses: A list of LLMResponse objects to cycle through. If None, a default
                       response is used.
        """
        if responses:
            self.responses = responses
        else:
            self.responses = [
                LLMResponse(
                    thinking="I will use the provided tools.",
                    code="print('Hello from Dummy')",
                )
            ]
        self.call_count = 0
        self.all_events: list[list[Event]] = []
        self.all_systems: list[str] = []

    def complete(self, system: str, events: List[Event], **kwargs) -> LLMResponse:
        """
        Return the next LLMResponse in the sequence, cycling through the list.
        If any event contains images, it prepends a note to the 'thinking' field.
        """
        # Store the received data for test inspection
        self.all_systems.append(system)
        self.all_events.append(events)

        # Get the next item in the cycle
        item = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        # If the item is an exception, raise it to simulate client failure
        if isinstance(item, Exception):
            raise item
        response = item.model_copy()

        # Check for any images in OutputEvent parts to simulate vision processing
        # Note: OutputEvent.parts contains raw objects, not ImagePart yet
        # The conversion to ImagePart happens in ContextRenderer during rendering
        has_images = False
        for event in events:
            if isinstance(event, OutputEvent):
                for part in event.parts:
                    # Check for common image types (PIL, matplotlib, numpy, etc.)
                    part_type = str(type(part))
                    if any(
                        img_type in part_type
                        for img_type in [
                            "PIL.Image",
                            "matplotlib.figure.Figure",
                            "plotly.graph_objs",
                            "numpy.ndarray",
                        ]
                    ):
                        has_images = True
                        break
                if has_images:
                    break

        if has_images:
            response.thinking = (
                f"[Dummy client acknowledges seeing an image.]\n{response.thinking}"
            )

        return response

    def summarize(self, system: str, content: str, **kwargs) -> str:
        """Return a deterministic plain text for testing."""
        # Simple concatenation for testing
        return f"{system} {content}".strip() or "dummy"

    @property
    def context_window(self) -> int:
        return 8192

    @property
    def model(self) -> str:
        return "dummy"

    @property
    def provider_name(self) -> str:
        return "Dummy"
