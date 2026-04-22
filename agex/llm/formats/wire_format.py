"""Wire-format abstraction for LLM communication.

A :class:`WireFormat` handles the format-specific concerns of an LLM
interaction:

1. Rendering the conversation event log into provider-agnostic message
   dicts (clients translate those to their provider's exact shape).
2. Parsing the provider's streaming response into ``TokenChunk``\\ s.
3. Supplying any format-specific system-prompt addendum.
4. Optionally declaring a tool schema for provider-native tool-calling.

Transport concerns (HTTP, auth, SSE framing, retries, streaming chunk
decode) remain with the client. Two implementations exist:

- :class:`~agex.llm.formats.xml.XmlWireFormat` — XML tags embedded in
  text content; single text stream; no provider tool-calling.
- :class:`~agex.llm.formats.tool_use.ToolUseWireFormat` — provider-native
  tool-calling; stream is a sequence of :class:`ToolCallEvent` objects.

Each concrete format supports ONE of the two parse paths:
``parse_text_stream`` (for XML) or ``parse_tool_stream`` (for
tool-use). The unsupported method raises :class:`NotImplementedError`.
Clients dispatch to the right one based on ``tool_schema()`` being
``None`` or not.

``WireFormat`` is a ``typing.Protocol`` rather than an ABC because it's
a pure interface — no shared behaviour to inherit. Implementers may
inherit for `isinstance` ergonomics, or just match the structure.
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
        format. May be an empty string for formats that rely on
        schema-level documentation (e.g. tool-use)."""
        ...

    def render_events(self, events: "list[Event]") -> "list[dict]":
        """Render the event log to provider-agnostic message dicts.

        Each message has a ``role`` and ``content``. ``content`` is
        either a plain string or a list of content parts (each with
        ``type`` = ``"text"`` or ``"image"``).

        Clients translate these dicts to their provider's specific
        shape (e.g. Anthropic content arrays, OpenAI ``tool_calls``).
        """
        ...

    def tool_schema(self) -> "list[dict] | None":
        """Provider-native tool schema, or ``None`` if this format
        doesn't use tool-calling."""
        ...

    def parse_text_stream(self, raw: Iterator[str]) -> Iterator["TokenChunk"]:
        """Parse a stream of raw text chunks into ``TokenChunk``\\ s.
        Used by formats whose provider response is a plain text stream
        (e.g. XML tags embedded in ``choices[].delta.content``).
        Tool-use formats raise :class:`NotImplementedError`."""
        ...

    def aparse_text_stream(
        self, raw: AsyncIterator[str]
    ) -> AsyncIterator["TokenChunk"]:
        """Async counterpart to :meth:`parse_text_stream`."""
        ...

    def parse_tool_stream(
        self, raw: Iterator["ToolCallEvent"]
    ) -> Iterator["TokenChunk"]:
        """Parse a stream of provider-agnostic tool-call events into
        ``TokenChunk``\\ s. Used by tool-use formats; text formats raise
        :class:`NotImplementedError`."""
        ...

    def aparse_tool_stream(
        self, raw: AsyncIterator["ToolCallEvent"]
    ) -> AsyncIterator["TokenChunk"]:
        """Async counterpart to :meth:`parse_tool_stream`."""
        ...
