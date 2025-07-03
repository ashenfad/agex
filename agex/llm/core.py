from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal

from pydantic import BaseModel


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


class LLMResponse(BaseModel):
    """Structured LLM response with parsed thinking and code sections."""

    thinking: str
    code: str


class ResponseParseError(Exception):
    """Exception raised when an agent's response cannot be parsed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class LLMClient(ABC):
    """
    A common interface for LLM clients, ensuring compatibility between different
    providers and implementation approaches.
    """

    @abstractmethod
    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Send messages to the LLM and get back a structured response.

        Args:
            messages: List of Message objects with role and content
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with parsed thinking and code sections

        Raises:
            RuntimeError: If the completion request fails
            ResponseParseError: If response doesn't match expected format
        """
        ...

    @abstractmethod
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
    @abstractmethod
    def context_window(self) -> int:
        """
        The maximum context window size for this model.

        Returns:
            Maximum number of tokens that can be sent in a single request
        """
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """
        The model name being used.

        Returns:
            Model identifier string
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        The provider name for this client.

        Returns:
            Provider name string (e.g., "OpenAI", "Anthropic", "Google Gemini")
        """
        ...
