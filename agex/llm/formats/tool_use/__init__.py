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
from .primer import TOOL_USE_FORMAT_PRIMER
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

    Text-stream parsing (:meth:`parse_text_stream` /
    :meth:`aparse_text_stream`) is not supported and raises
    :class:`NotImplementedError` — use :meth:`parse_tool_stream` /
    :meth:`aparse_tool_stream` instead.
    """

    def format_primer(self) -> str:
        return TOOL_USE_FORMAT_PRIMER

    def render_events(self, events: list[Event]) -> list[dict]:
        return render_events_as_tool_use(events)

    def tool_schema(self) -> list[dict] | None:
        return agex_tool_schemas()

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
    "TOOL_USE_FORMAT_PRIMER",
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
