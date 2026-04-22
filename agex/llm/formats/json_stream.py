"""Streaming JSON string-value extractor for tool-use wire formats.

Parses a streaming JSON object incrementally and yields deltas for each
top-level string value as its decoded content grows.  Non-string values
(numbers, booleans, ``null``, nested arrays/objects) are parsed and
skipped — no deltas are emitted for them.  This matches how agex plans
to shape tool-use arguments: a flat object whose streamable text fields
(``title``, ``thinking``, ``report``, ``code``, ``terminal``) are all
top-level strings.

Typical use: consume tool-argument deltas from a provider
(Anthropic ``input_json_delta``, OpenAI tool_call arguments) to stream
agent-facing text to the UI as it arrives, without waiting for the tool
call to close.
"""

from dataclasses import dataclass
from typing import AsyncIterator, Iterator


@dataclass(frozen=True, slots=True)
class JsonStringDelta:
    """One delta for a top-level string value in a streaming JSON object.

    ``done=True`` signals the value has finished; the ``content`` is
    empty on that delta.  Consumers accumulate the ``content`` fields
    until they receive a matching ``done`` delta for the same key.
    """

    key: str
    content: str
    done: bool


_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
}


# Parser states.
_BEFORE_OBJECT = 0
_EXPECT_KEY_OR_END = 1
_IN_KEY = 2
_EXPECT_COLON = 3
_EXPECT_VALUE = 4
_IN_STRING = 5
_SKIP_NON_STRING = 6
_EXPECT_COMMA_OR_END = 7
_DONE = 8


class JsonStringExtractor:
    """Incremental JSON parser that emits top-level string-value deltas.

    Call :meth:`feed` repeatedly with chunks of JSON text.  The parser
    tolerates chunk boundaries at any position, including mid-escape
    and mid-``\\uXXXX``.
    """

    __slots__ = (
        "_state",
        "_current_key",
        "_key_buf",
        "_value_buf",
        "_escape",
        "_unicode_hex",
        "_unicode_pending",
        "_skip_depth",
        "_skip_in_str",
        "_skip_esc",
    )

    def __init__(self) -> None:
        self._state = _BEFORE_OBJECT
        self._current_key = ""
        self._key_buf: list[str] = []
        self._value_buf: list[str] = []
        # Escape state inside string values.
        self._escape = False
        self._unicode_hex: list[str] = []
        self._unicode_pending = 0
        # State for skipping non-string values (numbers, bools, null,
        # nested arrays/objects).
        self._skip_depth = 0
        self._skip_in_str = False
        self._skip_esc = False

    def feed(self, chunk: str) -> Iterator[JsonStringDelta]:
        """Feed a chunk of JSON text; yield deltas as strings grow/close."""
        for ch in chunk:
            yield from self._consume(ch)
        # Flush any content buffered for the currently-open string.  Close
        # deltas are yielded inline (by _consume) the moment the closing
        # quote is seen, so this only fires when the chunk ends mid-value.
        if self._value_buf and self._state == _IN_STRING:
            yield JsonStringDelta(self._current_key, "".join(self._value_buf), False)
            self._value_buf.clear()

    def _consume(self, ch: str) -> Iterator[JsonStringDelta]:
        st = self._state

        if st == _BEFORE_OBJECT:
            if ch == "{":
                self._state = _EXPECT_KEY_OR_END
            # Ignore whitespace / leading garbage defensively.
            return

        if st == _EXPECT_KEY_OR_END:
            if ch.isspace():
                return
            if ch == '"':
                self._key_buf = []
                self._state = _IN_KEY
            elif ch == "}":
                self._state = _DONE
            return

        if st == _IN_KEY:
            if self._escape:
                self._key_buf.append(_SIMPLE_ESCAPES.get(ch, ch))
                self._escape = False
            elif ch == "\\":
                self._escape = True
            elif ch == '"':
                self._current_key = "".join(self._key_buf)
                self._state = _EXPECT_COLON
            else:
                self._key_buf.append(ch)
            return

        if st == _EXPECT_COLON:
            if ch == ":":
                self._state = _EXPECT_VALUE
            # Whitespace permitted between key and colon.
            return

        if st == _EXPECT_VALUE:
            if ch.isspace():
                return
            if ch == '"':
                self._state = _IN_STRING
                self._value_buf = []
                self._escape = False
                self._unicode_pending = 0
                return
            # Non-string value: skip until balanced close or comma.
            self._skip_in_str = False
            self._skip_esc = False
            self._skip_depth = 1 if ch in "{[" else 0
            self._state = _SKIP_NON_STRING
            return

        if st == _IN_STRING:
            if self._unicode_pending > 0:
                self._unicode_hex.append(ch)
                self._unicode_pending -= 1
                if self._unicode_pending == 0:
                    try:
                        code = int("".join(self._unicode_hex), 16)
                        self._value_buf.append(chr(code))
                    except ValueError:
                        # Malformed \u sequence; emit replacement char.
                        self._value_buf.append("\ufffd")
                    self._unicode_hex = []
                return
            if self._escape:
                self._escape = False
                if ch == "u":
                    self._unicode_pending = 4
                    self._unicode_hex = []
                else:
                    self._value_buf.append(_SIMPLE_ESCAPES.get(ch, ch))
                return
            if ch == "\\":
                self._escape = True
                return
            if ch == '"':
                # Flush accumulated content, then emit close delta.
                if self._value_buf:
                    yield JsonStringDelta(
                        self._current_key, "".join(self._value_buf), False
                    )
                    self._value_buf.clear()
                yield JsonStringDelta(self._current_key, "", True)
                self._state = _EXPECT_COMMA_OR_END
                return
            self._value_buf.append(ch)
            return

        if st == _SKIP_NON_STRING:
            if self._skip_in_str:
                if self._skip_esc:
                    self._skip_esc = False
                elif ch == "\\":
                    self._skip_esc = True
                elif ch == '"':
                    self._skip_in_str = False
                return
            if ch == '"':
                self._skip_in_str = True
                return
            if ch in "{[":
                self._skip_depth += 1
                return
            if ch in "}]":
                if self._skip_depth > 0:
                    self._skip_depth -= 1
                    if self._skip_depth == 0:
                        self._state = _EXPECT_COMMA_OR_END
                    return
                # Unmatched close — a bare literal (42, true, null) ended
                # at the object's closing brace.
                if ch == "}":
                    self._state = _DONE
                return
            if ch == "," and self._skip_depth == 0:
                self._state = _EXPECT_KEY_OR_END
                return
            return

        if st == _EXPECT_COMMA_OR_END:
            if ch.isspace():
                return
            if ch == ",":
                self._state = _EXPECT_KEY_OR_END
            elif ch == "}":
                self._state = _DONE
            return

        # _DONE: ignore further input.


def iter_json_strings(chunks: Iterator[str]) -> Iterator[JsonStringDelta]:
    """Extract top-level string-value deltas from a chunk iterator."""
    extractor = JsonStringExtractor()
    for chunk in chunks:
        yield from extractor.feed(chunk)


async def aiter_json_strings(
    chunks: AsyncIterator[str],
) -> AsyncIterator[JsonStringDelta]:
    """Async counterpart to :func:`iter_json_strings`."""
    extractor = JsonStringExtractor()
    async for chunk in chunks:
        for delta in extractor.feed(chunk):
            yield delta
