"""Wire-format implementations for LLM communication.

Each format converts between agex's event log and a provider's
on-the-wire shape, and parses the provider's response back into
``TokenChunk``\\ s.

Currently one format exists: :class:`ToolUseWireFormat` — provider-
native tool-calling (Anthropic, OpenAI, Gemini) that consumes a
provider-agnostic stream of :class:`ToolCallEvent`\\ s translated by
each client.  The :class:`WireFormat` protocol is kept as a seam for
hypothetical future wire formats (e.g. Responses API, local-model
grammar paths).
"""

from .tool_use import (
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallEvent,
    ToolCallStart,
    ToolUseWireFormat,
)
from .wire_format import WireFormat

__all__ = [
    "WireFormat",
    "ToolUseWireFormat",
    "ToolCallEvent",
    "ToolCallStart",
    "ToolCallArgDelta",
    "ToolCallEnd",
    "ThinkingPart",
]
