"""Wire-format implementations for LLM communication.

Each format converts between agex's event log and a provider's
on-the-wire shape, and parses the provider's response back into
``TokenChunk``\\ s.

Available formats:

- :class:`XmlWireFormat` — XML tags embedded in plain text.  Works with
  any provider returning a text stream; no tool-calling.
- :class:`ToolUseWireFormat` — provider-native tool-calling (Anthropic,
  OpenAI, Gemini).  Consumes a provider-agnostic stream of
  :class:`ToolCallEvent`\\ s translated by each client.
"""

from .tool_use import (
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallEvent,
    ToolCallStart,
    ToolUseWireFormat,
)
from .wire_format import WireFormat
from .xml import XmlWireFormat

__all__ = [
    "WireFormat",
    "XmlWireFormat",
    "ToolUseWireFormat",
    "ToolCallEvent",
    "ToolCallStart",
    "ToolCallArgDelta",
    "ToolCallEnd",
]
