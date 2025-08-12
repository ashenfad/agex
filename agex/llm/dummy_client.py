"""
Dummy LLM client for testing purposes.

This module provides a mock LLMClient that returns predefined LLMResponse objects
sequentially, useful for testing agent behavior without actual LLM calls.
"""

from typing import List

from .core import LLMClient, LLMResponse, Message, MultimodalMessage


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
        self.all_messages: list[list[Message]] = []

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Return the next LLMResponse in the sequence, cycling through the list.
        If any message is a MultimodalMessage, it prepends a note to the 'thinking' field.
        """
        # Store the received messages for test inspection
        self.all_messages.append(messages)

        # Get the next item in the cycle
        item = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        # If the item is an exception, raise it to simulate client failure
        if isinstance(item, Exception):
            raise item
        response = item.model_copy()

        # Check for any multimodal messages to simulate vision processing
        has_images = any(
            isinstance(msg, MultimodalMessage)
            and any(part.type == "image" for part in msg.content)
            for msg in messages
        )

        if has_images:
            response.thinking = (
                f"[Dummy client acknowledges seeing an image.]\n{response.thinking}"
            )

        return response

    @property
    def context_window(self) -> int:
        return 8192

    @property
    def model(self) -> str:
        return "dummy"

    @property
    def provider_name(self) -> str:
        return "Dummy"
