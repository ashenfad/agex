"""
Dummy LLM client for testing purposes.

This module provides a mock LLMClient that returns predefined LLMResponse objects
sequentially, useful for testing agent behavior without actual LLM calls.
"""

from typing import List

from .core import LLMClient, LLMResponse, Message


class DummyLLMClient(LLMClient):
    """
    A dummy LLM client that returns predefined LLMResponse objects in sequence.
    Useful for testing agent logic without actual LLM calls.
    """

    def __init__(self, responses: List[LLMResponse] | None = None):
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

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Return the next LLMResponse in the sequence, cycling through the list.
        """
        # Get the next response in the cycle
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    def estimate_tokens(self, text: str) -> int:
        # A rough estimate for testing purposes
        return len(text) // 4

    @property
    def context_window(self) -> int:
        return 8192

    @property
    def model(self) -> str:
        return "dummy"

    @property
    def provider_name(self) -> str:
        return "Dummy"
