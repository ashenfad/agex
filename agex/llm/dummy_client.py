"""
Dummy LLM client for testing purposes.

This module provides a mock LLMClient that returns predefined LLMResponse objects
sequentially, useful for testing agent behavior without actual LLM calls.
"""

from typing import List

from agex.agent.events import Event

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
        Exercises the same rendering path as real clients to catch image serialization issues.
        """
        # Store the received data for test inspection
        self.all_systems.append(system)
        self.all_events.append(events)

        # Exercise the same rendering path as real clients
        # This will call render_item_stream() and _serialize_image_to_base64()
        # allowing us to see what happens when images fail to serialize
        from agex.render.events import render_events_as_markdown

        max_tokens = kwargs.get("max_tokens", 4096)
        try:
            messages_dicts = render_events_as_markdown(events, self.model, max_tokens)
            # Store rendered messages for inspection
            self.all_rendered_messages = getattr(self, "all_rendered_messages", [])
            self.all_rendered_messages.append(messages_dicts)

            # Check for image export failures in rendered messages
            has_unsupported_images = False
            for msg in messages_dicts:
                content = msg.get("content", "")
                if isinstance(content, str) and (
                    "<unsupported image type:" in content
                    or "<image export failed:" in content
                ):
                    has_unsupported_images = True
                elif isinstance(content, list):
                    # Check multimodal content parts
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if (
                                "<unsupported image type:" in text
                                or "<image export failed:" in text
                            ):
                                has_unsupported_images = True
        except Exception:
            # Silently handle rendering errors in dummy client
            pass

        # Get the next item in the cycle
        item = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        # If the item is an exception, raise it to simulate client failure
        if isinstance(item, Exception):
            raise item
        response = item.model_copy()

        # If we detected unsupported images, note it in the response
        if has_unsupported_images:
            response.thinking = f"[Dummy client detected unsupported image type during rendering.]\n{response.thinking}"

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
