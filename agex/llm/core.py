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

from pydantic import BaseModel, Field

from agex.agent.datatypes import EditAction, FileAction

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from agex.agent.events import Event

# ============================================================================
# Timeout Configuration
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = 90.0  # 90 seconds per API call (used by LLM base)

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
        type: Either "title", "thinking", "python", "file", "edit", or "terminal"
        content: The text content (incremental)
        done: True when this section is complete
    """

    type: Literal["title", "thinking", "python", "file", "edit", "terminal"]
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
    """Structured LLM response with parsed title, thinking, code, and files sections."""

    title: str = ""
    thinking: str
    code: str | None = None
    file_actions: list[FileAction | EditAction] = Field(default_factory=list)
    terminal: str | None = None


class ResponseParseError(Exception):
    """Exception raised when an agent's response cannot be parsed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class ResponseBuilder:
    """Helper to accumulate tokens from a stream into an LLMResponse."""

    def __init__(
        self,
        agent_name: str | None = None,
        exec_state: "MutableMapping[str, Any] | None" = None,
    ):
        self.agent_name = agent_name
        self.exec_state = exec_state
        self.title_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.code_parts: list[str] = []
        self.terminal_parts: list[str] = []
        self.file_parts: dict[str, list[str]] = {}
        self.file_modes: dict[str, str] = {}
        self.current_file_path: str | None = None
        # Edit tracking
        self.edit_parts: dict[str, list[str]] = {}
        self.edit_metadata: dict[str, dict] = {}  # path -> {replace_all: bool}
        self.current_edit_path: str | None = None
        # Track ordering of file and edit actions
        self.action_order: list[tuple[str, str]] = []  # [(type, path), ...]
        self.seen_sections: dict[str, bool] = {
            "title": False,
            "thinking": False,
            "python": False,
            "file": False,
            "edit": False,
            "terminal": False,
        }

    def process_token(self, token: TokenChunk) -> StreamToken:
        """Process a raw TokenChunk and return an enriched StreamToken."""
        start_flag = (
            not token.done
            and token.type in self.seen_sections
            and not self.seen_sections[token.type]
        )
        if start_flag and token.type in self.seen_sections:
            if token.type not in ("file", "edit"):
                self.seen_sections[token.type] = True

        enriched = StreamToken(
            type=token.type,
            content=token.content,
            done=token.done,
            agent_name=self.agent_name or "",
            full_namespace=getattr(self.exec_state, "namespace", self.agent_name or ""),
            timestamp=datetime.now(timezone.utc),
            start=start_flag,
        )

        if token.done:
            if token.type == "file":
                self.current_file_path = None
            elif token.type == "edit":
                self.current_edit_path = None
            return enriched

        if token.type == "title":
            self.title_parts.append(token.content)
        elif token.type == "thinking":
            self.thinking_parts.append(token.content)
        elif token.type == "python":
            self.code_parts.append(token.content)
        elif token.type == "terminal":
            self.terminal_parts.append(token.content)
        elif token.type == "file":
            if token.content.startswith("path="):
                # Parse path and mode from metadata: "path=foo.py,mode=append"
                metadata = token.content
                import re

                from agex.llm.xml import validate_file_mode, validate_file_path

                path_match = re.search(r"path=([^,]+)", metadata)
                mode_match = re.search(r"mode=([^,]+)", metadata)

                if path_match:
                    path = validate_file_path(path_match.group(1))
                    mode_str = mode_match.group(1) if mode_match else "write"
                    mode = validate_file_mode(mode_str, path)
                    self.current_file_path = path
                    self.file_parts[self.current_file_path] = []
                    self.file_modes[self.current_file_path] = mode
                    # Track ordering
                    self.action_order.append(("file", path))
            elif self.current_file_path:
                self.file_parts[self.current_file_path].append(token.content)
        elif token.type == "edit":
            if token.content.startswith("path="):
                # Parse path and metadata: "path=foo.py,match_all=False"
                metadata = token.content
                import re

                from agex.llm.xml import validate_file_path

                path_match = re.search(r"path=([^,]+)", metadata)
                match_all_match = re.search(r"match_all=([^,]+)", metadata)

                if path_match:
                    path = validate_file_path(path_match.group(1))
                    match_all = (
                        match_all_match is not None
                        and match_all_match.group(1).lower() == "true"
                    )
                    self.current_edit_path = path
                    self.edit_parts[path] = []
                    self.edit_metadata[path] = {
                        "match_all": match_all,
                    }
                    # Track ordering
                    self.action_order.append(("edit", path))
            elif self.current_edit_path:
                self.edit_parts[self.current_edit_path].append(token.content)

        return enriched

    def build(self) -> LLMResponse:
        """Return the final LLMResponse."""
        import re

        from agex.agent.datatypes import EditAction, FileAction
        from agex.llm.xml import (
            TAG_INSERT_AFTER,
            TAG_INSERT_BEFORE,
            TAG_REPLACE,
            TAG_SEARCH,
            validate_edit_search,
        )

        file_actions: list[FileAction | EditAction] = []

        # Build actions in the order they appeared
        for action_type, path in self.action_order:
            if action_type == "file":
                parts = self.file_parts.get(path, [])
                content = "".join(parts)
                mode = self.file_modes.get(path, "write")
                file_actions.append(
                    FileAction(path=path, content=content, mode=mode)  # type: ignore[arg-type]
                )
            elif action_type == "edit":
                parts = self.edit_parts.get(path, [])
                inner_content = "".join(parts)
                metadata = self.edit_metadata.get(path, {})
                match_all = metadata.get("match_all", False)

                # Parse SEARCH tag (required)
                search_match = re.search(
                    rf"<{TAG_SEARCH}>(.*?)</{TAG_SEARCH}>",
                    inner_content,
                    re.DOTALL | re.IGNORECASE,
                )

                # Parse operation tag - REPLACE, INSERT-AFTER, or INSERT-BEFORE
                replace_match = re.search(
                    rf"<{TAG_REPLACE}>(.*?)</{TAG_REPLACE}>",
                    inner_content,
                    re.DOTALL | re.IGNORECASE,
                )
                insert_after_match = re.search(
                    rf"<{TAG_INSERT_AFTER}>(.*?)</{TAG_INSERT_AFTER}>",
                    inner_content,
                    re.DOTALL | re.IGNORECASE,
                )
                insert_before_match = re.search(
                    rf"<{TAG_INSERT_BEFORE}>(.*?)</{TAG_INSERT_BEFORE}>",
                    inner_content,
                    re.DOTALL | re.IGNORECASE,
                )

                # Determine operation and content
                if replace_match:
                    operation = "replace"
                    content = replace_match.group(1)
                elif insert_after_match:
                    operation = "insert-after"
                    content = insert_after_match.group(1)
                elif insert_before_match:
                    operation = "insert-before"
                    content = insert_before_match.group(1)
                else:
                    continue  # Skip if no operation tag found

                if search_match:
                    search = search_match.group(1)

                    validate_edit_search(path, search)
                    file_actions.append(
                        EditAction(
                            path=path,
                            search=search,
                            content=content,
                            operation=operation,
                            match_all=match_all,
                        )
                    )

        return LLMResponse(
            title="".join(self.title_parts).strip(),
            thinking="".join(self.thinking_parts),
            code="".join(self.code_parts),
            file_actions=file_actions,
            terminal="".join(self.terminal_parts) if self.terminal_parts else None,
        )


class LLM(ABC):
    """
    Abstract base class for LLM providers.

    Provides a common interface ensuring compatibility between different
    providers and implementation approaches.
    """

    def __init__(
        self,
        model: str = "",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        **kwargs,
    ):
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds for each API call. Override in subclass to customize."""
        return self._timeout_seconds

    def dump_config(self) -> dict[str, Any]:
        """
        Return a configuration dictionary that can be used to reconstruct this client.

        This dictionary should be serializable (JSON-safe).
        """
        return {
            "provider": getattr(self, "provider_name", "llmclient").lower(),
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLM":
        """
        Reconstruct an LLM from a configuration dictionary.

        This method delegates to the `connect_llm` factory.
        """
        # Avoid circular import
        from agex.llm import connect_llm

        return connect_llm(**config)

    def complete(self, system: str, events: list["Event"], **kwargs) -> LLMResponse:
        """
        Agent execution - convert events to structured response by consuming the stream.

        Args:
            system: System message content (primer + capabilities)
            events: Conversation history as Event objects
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with parsed thinking, code, and files sections
        """
        builder = ResponseBuilder()
        for token in self.complete_stream(system, events, **kwargs):
            builder.process_token(token)
        return builder.build()

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
        Async agent execution - convert events to structured response by consuming the stream.

        Args:
            system: System message content (primer + capabilities)
            events: Conversation history as Event objects
            **kwargs: Provider-specific arguments (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with parsed thinking, code, and files sections
        """
        builder = ResponseBuilder()
        async for token in self.acomplete_stream(system, events, **kwargs):
            builder.process_token(token)
        return builder.build()

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
        return getattr(self, "_model", "")

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        The provider name for this LLM.

        Returns:
            Provider name string (e.g., "OpenAI", "Anthropic", "Google Gemini")
        """
        ...
