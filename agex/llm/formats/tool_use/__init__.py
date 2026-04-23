"""Provider-native tool-use wire format.

Concrete :class:`~agex.llm.formats.wire_format.WireFormat` implementation
that expresses the agent's turn as a set of tool calls
(``python_action``, ``terminal_action``, ``write_file``, ``edit_file``)
rather than XML tags embedded in text.  Works with any provider that
supports function/tool calling (Anthropic, OpenAI, Gemini).
"""

from typing import AsyncIterator, Iterator

from agex.agent.events import Event
from agex.llm.core import TokenChunk

from .events import (
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallEvent,
    ToolCallStart,
)
from .parser import aparse_tool_events, parse_tool_events
from .primer import format_primer
from .renderer import render_events_as_tool_use
from .schemas import (
    ACTION_TOOLS,
    ALL_TOOLS,
    FILE_TOOLS,
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
    TOOL_TERMINAL,
    TOOL_WRITE_FILE,
    agex_tool_schemas,
)


class ToolUseWireFormat:
    """Tool-use wire format.

    Implements :class:`WireFormat` structurally.  Returns agex's tool
    schemas from :meth:`tool_schema`, renders history as tool_use /
    tool_result content blocks, and parses a provider-agnostic
    :class:`ToolCallEvent` stream into :class:`TokenChunk`\\ s.

    ``native_thinking`` strips the ``thinking`` and ``report``
    parameters from ``python_action`` / ``terminal_action`` schemas
    and appends a short primer addendum.  Flip it on when the
    provider delivers thinking and user-facing text as native content
    blocks (Gemini 3 thought parts; Claude 4.6 extended thinking;
    GPT-5 server-side reasoning) — the stream translators capture
    those blocks as :class:`ThinkingEmission` / :class:`TextEmission`
    directly, so narration-via-schema becomes redundant.  Leave it
    off for providers that aren't emitting native thinking (older
    Claude, OpenRouter-to-chat-class models, etc.).

    Text-stream parsing (:meth:`parse_text_stream` /
    :meth:`aparse_text_stream`) is not supported and raises
    :class:`NotImplementedError` — use :meth:`parse_tool_stream` /
    :meth:`aparse_tool_stream` instead.
    """

    def __init__(self, native_thinking: bool = False):
        self.native_thinking = native_thinking

    def format_primer(self) -> str:
        return format_primer(native_thinking=self.native_thinking)

    def render_events(self, events: list[Event]) -> list[dict]:
        return render_events_as_tool_use(events)

    def tool_schema(self) -> list[dict] | None:
        return agex_tool_schemas(native_thinking=self.native_thinking)

    def parse_text_stream(self, raw: Iterator[str]) -> Iterator[TokenChunk]:
        raise NotImplementedError(
            "ToolUseWireFormat parses tool-call events, not text streams. "
            "Use parse_tool_stream() instead."
        )

    def aparse_text_stream(self, raw: AsyncIterator[str]) -> AsyncIterator[TokenChunk]:
        raise NotImplementedError(
            "ToolUseWireFormat parses tool-call events, not text streams. "
            "Use aparse_tool_stream() instead."
        )

    def parse_tool_stream(self, raw: Iterator[ToolCallEvent]) -> Iterator[TokenChunk]:
        return parse_tool_events(raw)

    def aparse_tool_stream(
        self, raw: AsyncIterator[ToolCallEvent]
    ) -> AsyncIterator[TokenChunk]:
        return aparse_tool_events(raw)


__all__ = [
    "ToolUseWireFormat",
    "ToolCallEvent",
    "ToolCallStart",
    "ToolCallArgDelta",
    "ToolCallEnd",
    "ThinkingPart",
    "format_primer",
    "ACTION_TOOLS",
    "ALL_TOOLS",
    "FILE_TOOLS",
    "TOOL_EDIT_FILE",
    "TOOL_PYTHON",
    "TOOL_TERMINAL",
    "TOOL_WRITE_FILE",
    "agex_tool_schemas",
    "parse_tool_events",
    "aparse_tool_events",
    "render_events_as_tool_use",
]
