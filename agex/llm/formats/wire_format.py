"""Wire-format abstraction for LLM communication.

A :class:`WireFormat` handles the format-specific concerns of an LLM
interaction:

1. Rendering the conversation event log into provider-agnostic message
   dicts (clients translate those to their provider's exact shape).
2. Parsing the provider's streaming tool-call events into
   ``TokenChunk``\\ s.
3. Supplying any format-specific system-prompt addendum.
4. Declaring the tool schema for provider-native tool-calling.

Transport concerns (HTTP, auth, SSE framing, retries, streaming chunk
decode) remain with the client.  The single concrete implementation
is :class:`~agex.llm.formats.tool_use.ToolUseWireFormat` —
provider-native tool-calling, stream parsed as a sequence of
:class:`ToolCallEvent` objects.  The protocol is kept as a seam for
hypothetical future wire formats (e.g. Responses API, local-model
grammar paths).

``WireFormat`` is a ``typing.Protocol`` rather than an ABC because
it's a pure interface — no shared behaviour to inherit.  Implementers
may inherit for ``isinstance`` ergonomics, or just match the
structure.
"""

from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Iterator,
    runtime_checkable,
)
from typing import (
    Protocol as _TypingProtocol,
)

if TYPE_CHECKING:
    from agex.agent.events import Event
    from agex.llm.core import TokenChunk
    from agex.llm.formats.tool_use.events import ToolCallEvent


@runtime_checkable
class WireFormat(_TypingProtocol):
    """Format-specific rendering, stream parsing, and system-prompt
    contributions for an LLM interaction.
    """

    def format_primer(self) -> str:
        """Text to append to the system prompt describing the wire
        format.  May be an empty string for formats that rely on
        schema-level documentation alone."""
        ...

    def render_events(self, events: "list[Event]") -> "list[dict]":
        """Render the event log to provider-agnostic message dicts.

        Each message has a ``role`` and ``content``.  ``content`` is
        either a plain string or a list of content parts (each with
        ``type`` = ``"text"`` / ``"image"`` / ``"tool_use"`` /
        ``"tool_result"`` / ``"thinking"``).

        Clients translate these dicts to their provider's specific
        shape.
        """
        ...

    def tool_schema(self) -> "list[dict]":
        """Provider-native tool schema used for function-calling."""
        ...

    def parse_tool_stream(
        self, raw: Iterator["ToolCallEvent"]
    ) -> Iterator["TokenChunk"]:
        """Parse a stream of provider-agnostic tool-call events into
        ``TokenChunk``\\ s."""
        ...

    def aparse_tool_stream(
        self, raw: AsyncIterator["ToolCallEvent"]
    ) -> AsyncIterator["TokenChunk"]:
        """Async counterpart to :meth:`parse_tool_stream`."""
        ...
