import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Literal,
    TypeVar,
    Union,
)

from pydantic import BaseModel

if TYPE_CHECKING:
    from agex.agent.events import Event

# ============================================================================
# Timeout Configuration
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = 90.0  # 90 seconds per API call (used by LLMClient base)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_timeout(fn: Callable[[], T], timeout: float) -> T:
    """
    Execute a sync function with timeout.

    Args:
        fn: Zero-argument callable to execute
        timeout: Timeout in seconds

    Returns:
        Result of fn()

    Raises:
        TimeoutError: If the call times out
        Exception: Any exception from fn
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"Request timed out after {timeout}s")


async def with_timeout_async(coro_fn: Callable[[], Any], timeout: float) -> Any:
    """
    Execute an async coroutine with timeout.

    Args:
        coro_fn: Zero-argument callable that returns a coroutine
        timeout: Timeout in seconds

    Returns:
        Result of awaiting coro_fn()

    Raises:
        TimeoutError: If the call times out
        Exception: Any exception from coro_fn
    """
    try:
        return await asyncio.wait_for(coro_fn(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Request timed out after {timeout}s")


# ============================================================================
# Content Types
# ============================================================================


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

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds for each API call. Override in subclass to customize."""
        return DEFAULT_TIMEOUT_SECONDS

    @abstractmethod
    def complete(self, system: str, events: list["Event"], **kwargs) -> LLMResponse:
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
        self, system: str, events: list["Event"], **kwargs
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

    async def acomplete(
        self, system: str, events: list["Event"], **kwargs
    ) -> LLMResponse:
        """
        Async agent execution - convert events to structured response.

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
        # Default fallback: run sync complete() in a thread
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.complete(system, events, **kwargs)
        )

    async def acomplete_stream(
        self, system: str, events: list["Event"], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """
        Async agent execution with token-level streaming support.

        Args:
            system: System message content (primer + capabilities)
            events: Conversation history as Event objects
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Yields:
            TokenChunk objects as sections are parsed from the stream
        """
        # Default fallback: buffer acomplete() response and yield as tokens
        response = await self.acomplete(system, events, **kwargs)

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
        self, content: str | list["Event"]
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
    def summarize(self, system: str, content: str | list["Event"], **kwargs) -> str:
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
