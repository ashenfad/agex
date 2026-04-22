"""Wire-format implementations for LLM communication.

Each format converts between agex's event log and a provider's
on-the-wire shape, and parses the provider's response back into
``TokenChunk``\\ s.

Available formats:

- :class:`XmlWireFormat` — XML tags embedded in plain text. Works with
  every provider that returns a text stream; no tool-calling.
"""

from .wire_format import WireFormat
from .xml import XmlWireFormat

__all__ = ["WireFormat", "XmlWireFormat"]
