import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Literal,
    Union,
)

from pydantic import BaseModel, Field

from agex.agent.emissions import (
    Emission,
    FileEditEmission,
    FileWriteEmission,
    PythonEmission,
    TerminalEmission,
    TextEmission,
    ThinkingEmission,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from agex.agent.events import Event

# ============================================================================
# Timeout Configuration
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = 90.0  # 90 seconds per API call (used by LLM base)

logger = logging.getLogger(__name__)


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
    """A piece of streamed content from the LLM.

    Not an Event — tokens are ephemeral and don't go in the state log.

    One assistant turn contains one or more emissions.  Chunks are
    grouped by ``emission_index`` (starts at 0, increments each time a
    new emission begins in the stream).  The ``type`` tells the builder
    which field of the emission this chunk belongs to:

    - ``title`` / ``thinking`` — streamed narration for an action tool
      call.  Same ``emission_index`` as the action; ``thinking`` becomes
      a standalone :class:`ThinkingEmission` (inserted before the
      action) and ``title`` attaches to the action emission.
    - ``text`` — user-facing prose, a standalone :class:`TextEmission`.
    - ``python`` / ``terminal`` — code or shell commands for the main
      action emission at the same ``emission_index``.
    - ``file_path`` / ``file_search`` / ``file_content`` — streamed
      arg values for ``write_file`` / ``edit_file`` tool calls so the
      UI can show file-write progress as it happens.  These are
      UI-only; the authoritative :class:`FileWriteEmission` /
      :class:`FileEditEmission` still arrives in a terminal
      ``emission`` chunk at ``ToolCallEnd`` (it needs the full args
      JSON to resolve non-string fields like ``mode`` and
      ``match_all``).
    - ``emission`` — a fully-built :class:`Emission` object delivered
      in one shot via the ``emission`` field.  Used for file
      operations (``write_file`` / ``edit_file``) whose args are
      buffered by the parser and finalized together, and may be used
      by provider adapters to emit pre-built Text/Thinking emissions
      with signature metadata in Phase 4.
    - ``tool_start`` — a marker emitted by the parser on each action-
      tool call (``python_action`` / ``terminal_action``) carrying the
      tool name in ``content``.  The builder uses it to preserve the
      emission kind when the action was called with empty ``code`` /
      ``commands`` — without the marker, content-based inference
      would silently demote the call to a :class:`ThinkingEmission`,
      losing the fact that the model tried to call a tool.
    """

    type: Literal[
        "title",
        "thinking",
        "text",
        "python",
        "terminal",
        "file_path",
        "file_search",
        "file_content",
        "emission",
        "signature",
        "tool_start",
    ]
    content: str = ""
    done: bool = False
    emission_index: int = 0
    emission: Emission | None = None
    signature: bytes | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class StreamToken(TokenChunk):
    """TokenChunk enriched with agent metadata for on_token handlers."""

    agent_name: str = ""
    full_namespace: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start: bool = False


class LLMResponse(BaseModel):
    """Parsed assistant turn as an ordered list of emissions.

    The list preserves stream-arrival order so provider-native shapes
    (Claude interleaved thinking, Gemini per-call thought_signature)
    round-trip faithfully.
    """

    emissions: list[Emission] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None

    class Config:
        arbitrary_types_allowed = True


class ResponseParseError(Exception):
    """Exception raised when an agent's response cannot be parsed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class EmissionsBuilder:
    """Assemble streamed :class:`TokenChunk`\\ s into an
    :class:`LLMResponse`.

    Chunks are grouped by ``emission_index``.  Within a group:

    - ``emission`` typed chunks carry a fully-built :class:`Emission`
      (used by file tools and provider-native adapters) and pass
      through as-is.
    - ``title`` / ``thinking`` / ``python`` / ``terminal`` / ``text``
      chunks stream character-by-character and are concatenated when
      the group is flushed into its final emission type:

      =============================== ==============================
      tokens present                  resulting emission(s)
      =============================== ==============================
      ``thinking`` + ``python``       ThinkingEmission, PythonEmission
      ``thinking`` + ``terminal``     ThinkingEmission, TerminalEmission
      ``python``                      PythonEmission
      ``terminal``                    TerminalEmission
      ``text``                        TextEmission
      ``thinking`` only               ThinkingEmission
      =============================== ==============================

      ``title`` attaches to the Python/Terminal emission in the group
      (as the UI label).  If a group contains ``thinking`` plus
      ``python`` / ``terminal``, the narrated thinking becomes a
      separate :class:`ThinkingEmission` inserted immediately before
      the action emission — matching the shape native-thinking
      providers will deliver in Phase 4.
    """

    def __init__(
        self,
        agent_name: str | None = None,
        exec_state: "MutableMapping[str, Any] | None" = None,
    ):
        self.agent_name = agent_name
        self.exec_state = exec_state
        # emission_index -> accumulator dict (keys: type -> list[str], or
        # "_prebuilt" -> Emission)
        self._slots: dict[int, dict[str, Any]] = {}
        # Track seen (emission_index, type) pairs so ``start`` flag can
        # flip on the first streamed chunk of each new field.
        self._seen: set[tuple[int, str]] = set()
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def process_token(self, token: TokenChunk) -> StreamToken:
        """Accumulate a chunk and return an enriched :class:`StreamToken`
        for UI callbacks.
        """
        # Capture usage data if present on this token
        if token.input_tokens is not None:
            self.input_tokens = token.input_tokens
        if token.output_tokens is not None:
            self.output_tokens = token.output_tokens

        key = (token.emission_index, token.type)
        start_flag = False
        if not token.done and token.type != "emission":
            if key not in self._seen:
                start_flag = True
                self._seen.add(key)

        enriched = StreamToken(
            type=token.type,
            content=token.content,
            done=token.done,
            emission_index=token.emission_index,
            emission=token.emission,
            signature=token.signature,
            input_tokens=token.input_tokens,
            output_tokens=token.output_tokens,
            agent_name=self.agent_name or "",
            full_namespace=getattr(self.exec_state, "namespace", self.agent_name or ""),
            timestamp=datetime.now(timezone.utc),
            start=start_flag,
        )

        slot = self._slots.setdefault(token.emission_index, {})

        if token.type == "emission":
            if token.emission is not None:
                slot["_prebuilt"] = token.emission
            return enriched

        if token.type == "signature":
            # Carried on its own token so it can arrive before the
            # content tokens stream in (Gemini sends the signature at
            # ToolCallStart time).
            if token.signature is not None:
                slot["_signature"] = token.signature
            return enriched

        if token.type == "tool_start":
            # Parser-emitted marker naming the tool at ToolCallStart.
            # Stashed here so ``build()`` can preserve the emission
            # kind even when the content args are empty (the model
            # tried to call a tool — we shouldn't demote it to a
            # ThinkingEmission just because ``code`` came in blank).
            slot["_tool"] = token.content
            return enriched

        # file_* tokens stream the write_file/edit_file args purely for
        # UI visibility.  The authoritative emission still arrives as a
        # prebuilt ``emission`` chunk at ToolCallEnd, so don't
        # accumulate here.
        if token.type in ("file_path", "file_search", "file_content"):
            return enriched

        # Skip the boundary markers; only accumulate content chunks.
        if not token.done:
            slot.setdefault(token.type, []).append(token.content)

        return enriched

    def build(self) -> LLMResponse:
        """Flush accumulated chunks into an ordered emission list."""
        emissions: list[Emission] = []
        for idx in sorted(self._slots.keys()):
            slot = self._slots[idx]

            prebuilt = slot.get("_prebuilt")
            signature = slot.get("_signature")
            if prebuilt is not None:
                if signature is not None and hasattr(prebuilt, "signature"):
                    prebuilt.signature = signature
                emissions.append(prebuilt)
                continue

            title_str = "".join(slot.get("title", [])).strip() or None
            thinking_str = "".join(slot.get("thinking", [])) or None
            python_str = "".join(slot.get("python", []))
            terminal_str = "".join(slot.get("terminal", []))
            text_str = "".join(slot.get("text", []))
            tool_marker = slot.get("_tool")

            # When the parser saw an action-tool call (python_action /
            # terminal_action), preserve the emission kind even if the
            # content arg ended up empty — the model *tried* to call
            # a tool and the loop will treat the empty code/commands as
            # a no-op.  Without this branch we'd silently demote to a
            # ThinkingEmission and the no-progress nudge would fire with
            # "plain text doesn't execute anything", which reads as
            # nonsense to a model that just called a tool.
            if tool_marker == "python_action":
                emissions.append(
                    PythonEmission(
                        code=python_str,
                        title=title_str,
                        thinking=thinking_str,
                        signature=signature,
                    )
                )
            elif tool_marker == "terminal_action":
                emissions.append(
                    TerminalEmission(
                        commands=terminal_str,
                        title=title_str,
                        thinking=thinking_str,
                        signature=signature,
                    )
                )
            # Narration-via-schema thinking rides on the action emission
            # for round-trip fidelity.  Standalone thinking (no
            # accompanying code/commands/text) becomes its own
            # :class:`ThinkingEmission` — that's the shape native-
            # thinking providers deliver.
            elif python_str:
                emissions.append(
                    PythonEmission(
                        code=python_str,
                        title=title_str,
                        thinking=thinking_str,
                        signature=signature,
                    )
                )
            elif terminal_str:
                emissions.append(
                    TerminalEmission(
                        commands=terminal_str,
                        title=title_str,
                        thinking=thinking_str,
                        signature=signature,
                    )
                )
            elif text_str:
                emissions.append(TextEmission(text=text_str, signature=signature))
                if thinking_str:
                    emissions.append(ThinkingEmission(text=thinking_str))
            elif thinking_str:
                emissions.append(
                    ThinkingEmission(text=thinking_str, signature=signature)
                )

        return LLMResponse(
            emissions=emissions,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
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
            LLMResponse with the emission list parsed from the stream
        """
        builder = EmissionsBuilder()
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

        Default implementation: Falls back to complete() and yields
        buffered emissions as a synthesized token stream.
        """
        response = self.complete(system, events, **kwargs)
        yield from _emissions_to_tokens(response)

    async def acomplete(
        self, system: str, events: list["Event"], **kwargs
    ) -> LLMResponse:
        """
        Async agent execution - convert events to structured response by consuming the stream.
        """
        builder = EmissionsBuilder()
        async for token in self.acomplete_stream(system, events, **kwargs):
            builder.process_token(token)
        return builder.build()

    async def acomplete_stream(
        self, system: str, events: list["Event"], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """Async counterpart to :meth:`complete_stream`.

        Default implementation: buffer :meth:`acomplete` and replay as a
        synthesized token stream.
        """
        response = await self.acomplete(system, events, **kwargs)
        for token in _emissions_to_tokens(response):
            yield token

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


def _emissions_to_tokens(response: LLMResponse) -> Iterator[TokenChunk]:
    """Synthesize a token stream from a buffered :class:`LLMResponse`.

    Used by the default :meth:`LLM.complete_stream` fallback for
    clients that only implement :meth:`complete`.  Order matches the
    original emission list; each streamed emission carries its own
    ``emission_index``.
    """
    for i, em in enumerate(response.emissions):
        if isinstance(em, TextEmission):
            if em.text:
                yield TokenChunk(
                    type="text", content=em.text, done=False, emission_index=i
                )
            yield TokenChunk(type="text", content="", done=True, emission_index=i)
        elif isinstance(em, ThinkingEmission):
            if em.text:
                yield TokenChunk(
                    type="thinking", content=em.text, done=False, emission_index=i
                )
            yield TokenChunk(type="thinking", content="", done=True, emission_index=i)
        elif isinstance(em, PythonEmission):
            if em.title:
                yield TokenChunk(
                    type="title", content=em.title, done=False, emission_index=i
                )
                yield TokenChunk(type="title", content="", done=True, emission_index=i)
            if em.thinking:
                yield TokenChunk(
                    type="thinking",
                    content=em.thinking,
                    done=False,
                    emission_index=i,
                )
                yield TokenChunk(
                    type="thinking", content="", done=True, emission_index=i
                )
            if em.code:
                yield TokenChunk(
                    type="python", content=em.code, done=False, emission_index=i
                )
            yield TokenChunk(type="python", content="", done=True, emission_index=i)
        elif isinstance(em, TerminalEmission):
            if em.title:
                yield TokenChunk(
                    type="title", content=em.title, done=False, emission_index=i
                )
                yield TokenChunk(type="title", content="", done=True, emission_index=i)
            if em.thinking:
                yield TokenChunk(
                    type="thinking",
                    content=em.thinking,
                    done=False,
                    emission_index=i,
                )
                yield TokenChunk(
                    type="thinking", content="", done=True, emission_index=i
                )
            if em.commands:
                yield TokenChunk(
                    type="terminal", content=em.commands, done=False, emission_index=i
                )
            yield TokenChunk(type="terminal", content="", done=True, emission_index=i)
        elif isinstance(em, FileWriteEmission):
            if em.path:
                yield TokenChunk(
                    type="file_path", content=em.path, done=False, emission_index=i
                )
            yield TokenChunk(type="file_path", content="", done=True, emission_index=i)
            if em.content:
                yield TokenChunk(
                    type="file_content",
                    content=em.content,
                    done=False,
                    emission_index=i,
                )
            yield TokenChunk(
                type="file_content", content="", done=True, emission_index=i
            )
            yield TokenChunk(
                type="emission",
                content="",
                done=True,
                emission_index=i,
                emission=em,
            )
        elif isinstance(em, FileEditEmission):
            if em.path:
                yield TokenChunk(
                    type="file_path", content=em.path, done=False, emission_index=i
                )
            yield TokenChunk(type="file_path", content="", done=True, emission_index=i)
            if em.search:
                yield TokenChunk(
                    type="file_search",
                    content=em.search,
                    done=False,
                    emission_index=i,
                )
            yield TokenChunk(
                type="file_search", content="", done=True, emission_index=i
            )
            if em.content:
                yield TokenChunk(
                    type="file_content",
                    content=em.content,
                    done=False,
                    emission_index=i,
                )
            yield TokenChunk(
                type="file_content", content="", done=True, emission_index=i
            )
            yield TokenChunk(
                type="emission",
                content="",
                done=True,
                emission_index=i,
                emission=em,
            )
