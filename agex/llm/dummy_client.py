"""
Dummy LLM client for testing purposes.

This module provides a mock LLMClient that returns predefined responses
sequentially, useful for testing agent behavior without actual LLM calls.
"""

import re
from typing import List

from .core import LLMClient, LLMResponse, Message


class DummyLLMClient(LLMClient):
    """
    A dummy LLM client that returns a fixed response or cycles through a list of responses.
    Useful for testing agent logic without actual LLM calls.
    """

    def __init__(self, responses: List[str] | None = None):
        """
        Initialize with a sequence of responses to return.

        Args:
            responses: A list of response strings to cycle through. If None, a default
                       response is used.
        """
        if responses:
            self.responses = responses
        else:
            self.responses = [
                "# Thinking\nI will use the provided tools.\n\n"
                "```python\nprint('Hello from Dummy')\n```"
            ]
        self.call_count = 0

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Return a fixed, structured LLMResponse, cycling through the list if provided.
        """
        # Get the next response in the cycle.
        response_text = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        # The thinking/code parts are extracted to mimic the real client's output.
        thinking_match = re.search(r"# Thinking\n(.*?)\n\n", response_text, re.DOTALL)
        code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)

        thinking = thinking_match.group(1).strip() if thinking_match else ""
        code = code_match.group(1).strip() if code_match else ""

        if not thinking and not code:
            if "python" not in response_text:
                code = response_text

        return LLMResponse(thinking=thinking, code=code)

    def estimate_tokens(self, text: str) -> int:
        # A rough estimate for testing purposes
        return len(text) // 4

    @property
    def context_window(self) -> int:
        return 8192

    @property
    def model(self) -> str:
        return "dummy"
