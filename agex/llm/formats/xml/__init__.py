"""XML wire format.

Concrete :class:`~agex.llm.formats.wire_format.WireFormat` implementation
that uses uppercase XML tags embedded in the provider's text stream.
This is the classic/legacy agex wire format — no provider-native tool
calling.
"""

from typing import TYPE_CHECKING, AsyncIterator, Iterator

from agex.agent.events import Event
from agex.llm.core import TokenChunk

if TYPE_CHECKING:
    from agex.llm.formats.tool_use.events import ToolCallEvent

from .renderer import render_events_as_xml
from .tags import (
    TAG_CANCELLED,
    TAG_CLARIFY,
    TAG_EDIT,
    TAG_FAIL,
    TAG_FILE,
    TAG_INSERT_AFTER,
    TAG_INSERT_BEFORE,
    TAG_OBSERVATION,
    TAG_PYTHON,
    TAG_REPLACE,
    TAG_REPORT,
    TAG_SEARCH,
    TAG_SUCCESS,
    TAG_TERMINAL,
    TAG_THINKING,
    TAG_TITLE,
    VALID_FILE_MODES,
    VALID_OPERATIONS,
    XML_FORMAT_PRIMER,
)
from .tokenizer import (
    XMLResponse,
    atokenize_xml_stream,
    parse_xml_response,
    tokenize_xml_stream,
)
from .validation import (
    validate_edit_search,
    validate_file_mode,
    validate_file_path,
)


class XmlWireFormat:
    """XML-in-text wire format.

    Implements :class:`WireFormat` structurally. Uses the XML primer for
    system-prompt instructions, renders events with uppercase XML tags,
    parses the provider's plain-text response into ``TokenChunk``\\ s,
    and declares no provider-native tool schema.
    """

    def format_primer(self) -> str:
        return XML_FORMAT_PRIMER

    def render_events(self, events: list[Event]) -> list[dict]:
        return render_events_as_xml(events)

    def tool_schema(self) -> list[dict] | None:
        return None

    def parse_text_stream(self, raw: Iterator[str]) -> Iterator[TokenChunk]:
        return tokenize_xml_stream(raw)

    def aparse_text_stream(self, raw: AsyncIterator[str]) -> AsyncIterator[TokenChunk]:
        return atokenize_xml_stream(raw)

    def parse_tool_stream(self, raw: "Iterator[ToolCallEvent]") -> Iterator[TokenChunk]:
        raise NotImplementedError(
            "XmlWireFormat parses text streams, not tool-call events. "
            "Use parse_text_stream() instead."
        )

    def aparse_tool_stream(
        self, raw: "AsyncIterator[ToolCallEvent]"
    ) -> AsyncIterator[TokenChunk]:
        raise NotImplementedError(
            "XmlWireFormat parses text streams, not tool-call events. "
            "Use aparse_text_stream() instead."
        )


__all__ = [
    "XmlWireFormat",
    "XMLResponse",
    "XML_FORMAT_PRIMER",
    "VALID_FILE_MODES",
    "VALID_OPERATIONS",
    "TAG_CANCELLED",
    "TAG_CLARIFY",
    "TAG_EDIT",
    "TAG_FAIL",
    "TAG_FILE",
    "TAG_INSERT_AFTER",
    "TAG_INSERT_BEFORE",
    "TAG_OBSERVATION",
    "TAG_PYTHON",
    "TAG_REPLACE",
    "TAG_REPORT",
    "TAG_SEARCH",
    "TAG_SUCCESS",
    "TAG_TERMINAL",
    "TAG_THINKING",
    "TAG_TITLE",
    "atokenize_xml_stream",
    "parse_xml_response",
    "render_events_as_xml",
    "tokenize_xml_stream",
    "validate_edit_search",
    "validate_file_mode",
    "validate_file_path",
]
