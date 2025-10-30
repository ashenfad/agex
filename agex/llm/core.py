from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    from agex.agent.events import Event


@dataclass
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ImagePart:
    """Represents a base64 encoded image."""

    image: str
    type: Literal["image"] = "image"


ContentPart = Union[TextPart, ImagePart]


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
    def complete(self, system: str, events: List["Event"], **kwargs) -> LLMResponse:
        """
        Agent execution - convert events to structured response.

        Args:
            system: System message content (primer + capabilities)
            events: Conversation history as Event objects
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with parsed thinking and code sections

        Raises:
            RuntimeError: If the completion request fails
            ResponseParseError: If response doesn't match expected format
        """
        ...

    @abstractmethod
    def summarize(self, system: str, content: str, **kwargs) -> str:
        """
        Generic text generation with instructions.

        Used for capabilities summarization and other non-agent tasks.
        Does not handle events or multimodal content.

        Args:
            system: Instructions for the task
            content: Text content to process
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Returns:
            Generated text string
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
