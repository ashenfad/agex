from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator, List, Literal, Union

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


@dataclass
class TokenChunk:
    """
    A piece of streamed content from the LLM.

    Not an Event - tokens are ephemeral and don't go in the state log.

    Attributes:
        type: Either "title", "thinking", or "python"
        content: The text content (incremental)
        done: True when this section is complete
    """

    type: Literal["title", "thinking", "python"]
    content: str
    done: bool = False


@dataclass
class StreamToken(TokenChunk):
    """TokenChunk enriched with agent metadata for on_token handlers."""

    agent_name: str = ""
    full_namespace: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start: bool = False


class LLMResponse(BaseModel):
    """Structured LLM response with parsed title, thinking, and code sections."""

    title: str = ""
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

    def complete_stream(
        self, system: str, events: List["Event"], **kwargs
    ) -> Iterator[TokenChunk]:
        """
        Agent execution with token-level streaming support.

        This method enables real-time UI feedback by yielding tokens as they arrive.
        Implementations can choose to support streaming or raise NotImplementedError.

        Default implementation: Falls back to complete() and yields buffered response.
        Providers that support streaming should override this method.

        Args:
            system: System message content (primer + capabilities)
            events: Conversation history as Event objects
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Yields:
            TokenChunk objects as sections are parsed from the stream

        Raises:
            NotImplementedError: If streaming is not supported by this client
            RuntimeError: If the completion request fails
            ResponseParseError: If response doesn't match expected format
        """
        # Default fallback: buffer complete() response and yield as tokens
        response = self.complete(system, events, **kwargs)

        # Yield title section first (if present)
        if response.title:
            yield TokenChunk(type="title", content=response.title, done=False)
            yield TokenChunk(type="title", content="", done=True)

        # Yield thinking section
        if response.thinking:
            yield TokenChunk(type="thinking", content=response.thinking, done=False)
        yield TokenChunk(type="thinking", content="", done=True)

        # Yield code section
        if response.code:
            yield TokenChunk(type="python", content=response.code, done=False)
        yield TokenChunk(type="python", content="", done=True)

    def _prepare_summarization_content(
        self, content: str | List["Event"]
    ) -> tuple[bool, Any]:
        """
        Helper to prepare content for summarization.

        Returns:
            (is_multimodal, processed_content)
            - If text: (False, text_string)
            - If events: (True, conversation_transcript_as_string)
        """
        if isinstance(content, list):
            # Import here to avoid circular dependency
            from agex.render.events import render_events_as_markdown

            messages = render_events_as_markdown(content)

            # Format as a transcript for summarization
            # Instead of sending alternating user/assistant messages (confusing),
            # send the entire conversation as a single text block to summarize
            transcript_parts = []
            for msg in messages:
                role = msg.get("role", "unknown").upper()
                content_value = msg.get("content", "")

                # Handle both string and list content
                if isinstance(content_value, list):
                    # Extract text from content parts
                    text_parts = []
                    for part in content_value:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    content_value = "\n".join(text_parts)

                transcript_parts.append(f"[{role}]:\n{content_value}\n")

            transcript = "\n".join(transcript_parts)
            framed_content = f"""You are an external observer summarizing a completed interaction. 
DO NOT respond as if you are the agent in this conversation.
DO NOT continue the conversation or take actions.

Below is the HISTORICAL TRANSCRIPT to summarize:

---BEGIN TRANSCRIPT---
{transcript}
---END TRANSCRIPT---

Write your summary of what happened in this interaction."""

            # Return as text (False) since we've converted it to a transcript
            return (False, framed_content)
        else:
            return (False, content)

    @abstractmethod
    def summarize(self, system: str, content: str | List["Event"], **kwargs) -> str:
        """
        Generic text generation with instructions.

        Used for capabilities summarization and event log summarization.
        Supports both plain text and events (with multimodal content).

        Args:
            system: Instructions for the task
            content: Either plain text OR list of events (may include images)
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Returns:
            Generated summary text
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
