from dataclasses import dataclass
from typing import List, Literal, Protocol


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


class LLMClient(Protocol):
    """
    A common interface for LLM clients, ensuring compatibility between different
    providers and implementation approaches.
    """

    def complete(self, messages: List[Message], **kwargs) -> str:
        """
        Send messages to the LLM and get back the response content as a string.

        Args:
            messages: List of Message objects with role and content
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Returns:
            The LLM's response content as a string

        Raises:
            RuntimeError: If the completion request fails
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        ...

    @property
    def context_window(self) -> int:
        """
        The maximum context window size for this model.

        Returns:
            Maximum number of tokens that can be sent in a single request
        """
        ...

    @property
    def model(self) -> str:
        """
        The model name being used.

        Returns:
            Model identifier string
        """
        ...
