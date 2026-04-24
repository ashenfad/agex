"""Convert a stream of provider-agnostic :class:`ToolCallEvent`\\ s into
:class:`TokenChunk`\\ s the agex :class:`EmissionsBuilder` consumes.

Each ``ToolCallStart`` bumps a per-turn ``emission_index`` counter so
the builder can group chunks by emission even when multiple tool calls
stream interleaved deltas.

Two cadences:

* **Action tools** (``python_action`` / ``terminal_action``) stream
  through in real time — each JSON string key (``title``, ``thinking``,
  ``code``/``commands``, ``report``) emits content ``TokenChunk``\\ s
  as characters arrive and a closing ``TokenChunk`` when the value
  finishes.  All chunks for a given call share its ``emission_index``.

* **File tools** (``write_file`` / ``edit_file``) stream their string
  args (``path``, ``content`` / ``search`` / ``replace``) as
  ``file_path`` / ``file_search`` / ``file_content``
  ``TokenChunk``\\ s so callers watching the stream can see what the
  model is writing.  The same
  bytes are also buffered so that at ``ToolCallEnd`` the parser can
  decode the full JSON (including non-string fields like ``mode`` and
  ``match_all``) and emit the authoritative
  ``TokenChunk(type="emission", emission=<FileWriteEmission |
  FileEditEmission>, done=True)`` that the :class:`EmissionsBuilder`
  slots into the final :class:`LLMResponse`.
"""

import json
from typing import AsyncIterator, Iterator

from agex.agent.emissions import (
    FileEditEmission,
    FileWriteEmission,
    TextEmission,
    ThinkingEmission,
)
from agex.llm.core import TokenChunk
from agex.llm.formats.json_stream import JsonStringExtractor

from .events import (
    TextPart,
    ThinkingPart,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallEvent,
    ToolCallStart,
)
from .schemas import (
    ACTION_TOOLS,
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
    TOOL_TERMINAL,
    TOOL_WRITE_FILE,
)

# JSON-schema key → TokenChunk.type for action tools.  ``report`` was
# the old user-facing-prose parameter; route it to the new ``text``
# token type so the builder produces a :class:`TextEmission`.  The
# schema field itself gets removed in Phase 3 along with the primer
# slim-down.
_PYTHON_KEY_MAP = {
    "title": "title",
    "thinking": "thinking",
    "report": "text",
    "code": "python",
}

_TERMINAL_KEY_MAP = {
    "title": "title",
    "thinking": "thinking",
    "report": "text",
    "commands": "terminal",
}

# File tool string args stream as UI-only ``file_*`` tokens.
_WRITE_FILE_KEY_MAP = {
    "path": "file_path",
    "content": "file_content",
}

_EDIT_FILE_KEY_MAP = {
    "path": "file_path",
    "search": "file_search",
    "replace": "file_content",
}


class _CallState:
    """Per-tool-call state: streaming extractor plus (for file tools) a
    raw buffer so the full JSON can be re-parsed at finalize time.

    ``emission_index`` is assigned at :class:`ToolCallStart` time from a
    monotonic counter owned by the parser; all tokens emitted for this
    call carry that same index.  ``signature`` is forwarded verbatim
    onto the built emission via the :class:`EmissionsBuilder`.
    """

    __slots__ = (
        "tool_name",
        "emission_index",
        "signature",
        "_extractor",
        "_key_map",
        "_raw_buf",
    )

    def __init__(
        self,
        tool_name: str,
        emission_index: int,
        signature: bytes | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.emission_index = emission_index
        self.signature = signature
        self._extractor = JsonStringExtractor()
        if tool_name == TOOL_PYTHON:
            self._key_map: dict[str, str] = _PYTHON_KEY_MAP
        elif tool_name == TOOL_TERMINAL:
            self._key_map = _TERMINAL_KEY_MAP
        elif tool_name == TOOL_WRITE_FILE:
            self._key_map = _WRITE_FILE_KEY_MAP
        elif tool_name == TOOL_EDIT_FILE:
            self._key_map = _EDIT_FILE_KEY_MAP
        else:
            self._key_map = {}
        # File tools also need the raw JSON at finalize time so
        # non-string fields (mode, match_all) survive into the built
        # emission.
        self._raw_buf: list[str] | None = (
            [] if tool_name in (TOOL_WRITE_FILE, TOOL_EDIT_FILE) else None
        )

    def feed_args(self, chunk: str) -> Iterator[TokenChunk]:
        if self._raw_buf is not None:
            self._raw_buf.append(chunk)
        for delta in self._extractor.feed(chunk):
            token_type = self._key_map.get(delta.key)
            if token_type is None:
                continue
            yield TokenChunk(
                type=token_type,
                content=delta.content,
                done=delta.done,
                emission_index=self.emission_index,
            )

    def finalize(self) -> Iterator[TokenChunk]:
        if self._raw_buf is None:
            # Action tool — streaming already yielded everything.
            return
        raw = "".join(self._raw_buf)
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            # Model produced invalid JSON for a file tool — drop it.
            # A higher-level retry will surface the error to the agent.
            return
        if not isinstance(args, dict):
            return
        emission = None
        if self.tool_name == TOOL_WRITE_FILE:
            emission = _build_write_file(args)
        elif self.tool_name == TOOL_EDIT_FILE:
            emission = _build_edit_file(args)
        if emission is not None:
            if self.signature is not None:
                emission.signature = self.signature
            yield TokenChunk(
                type="emission",
                content="",
                done=True,
                emission_index=self.emission_index,
                emission=emission,
            )


def _build_write_file(args: dict) -> FileWriteEmission | None:
    path = args.get("path") or ""
    if not path:
        return None
    content = args.get("content") or ""
    mode = args.get("mode") or "write"
    if mode not in ("write", "append"):
        mode = "write"
    return FileWriteEmission(path=path, content=content, mode=mode)  # type: ignore[arg-type]


def _build_edit_file(args: dict) -> FileEditEmission | None:
    path = args.get("path") or ""
    search = args.get("search")
    if not path or search is None or "replace" not in args:
        return None
    return FileEditEmission(
        path=path,
        search=search,
        content=args["replace"],
        match_all=bool(args.get("match_all", False)),
    )


class _ParserState:
    """Tracks open tool calls and assigns monotonic emission indices."""

    def __init__(self) -> None:
        self._calls: dict[str, _CallState] = {}
        self._next_index: int = 0

    def handle(self, event: ToolCallEvent) -> Iterator[TokenChunk]:
        if isinstance(event, TextPart):
            # Plain assistant text (not thinking, not a tool call).
            # Give it its own emission_index and deliver as a prebuilt
            # :class:`TextEmission`.  Whitespace-only text is noise
            # (providers occasionally emit a lone newline between
            # parts) — drop it so it doesn't clutter the event log.
            if not event.text or not event.text.strip():
                return
            idx = self._next_index
            self._next_index += 1
            yield TokenChunk(
                type="emission",
                content="",
                done=True,
                emission_index=idx,
                emission=TextEmission(text=event.text),
            )
            return
        if isinstance(event, ThinkingPart):
            # Native-thinking providers may emit signed thought parts
            # that must round-trip at their original position.  Give
            # them their own emission_index so the renderer can place
            # them among tool calls in the same order the model
            # produced them.
            if event.signature is None and not event.text and not event.redacted:
                return
            idx = self._next_index
            self._next_index += 1
            yield TokenChunk(
                type="emission",
                content="",
                done=True,
                emission_index=idx,
                emission=ThinkingEmission(
                    text=event.text or "",
                    signature=event.signature,
                    redacted=event.redacted,
                ),
            )
            return
        if isinstance(event, ToolCallStart):
            idx = self._next_index
            self._next_index += 1
            self._calls[event.call_id] = _CallState(
                event.tool_name, idx, event.signature
            )
            # Hoist the signature onto its own token so the builder can
            # slot it independently of the arg-delta stream.  File
            # tools get the signature applied inline at finalize() time,
            # but action tools need it stashed in the builder's slot
            # before their chunks stream in.
            if event.signature is not None and event.tool_name in ACTION_TOOLS:
                yield TokenChunk(
                    type="signature",
                    content="",
                    done=True,
                    emission_index=idx,
                    signature=event.signature,
                )
        elif isinstance(event, ToolCallArgDelta):
            state = self._calls.get(event.call_id)
            if state is not None:
                yield from state.feed_args(event.argument_chunk)
        elif isinstance(event, ToolCallEnd):
            state = self._calls.pop(event.call_id, None)
            if state is not None:
                yield from state.finalize()


def parse_tool_events(events: Iterator[ToolCallEvent]) -> Iterator[TokenChunk]:
    """Convert a synchronous tool-call event stream into TokenChunks."""
    state = _ParserState()
    for event in events:
        yield from state.handle(event)


async def aparse_tool_events(
    events: AsyncIterator[ToolCallEvent],
) -> AsyncIterator[TokenChunk]:
    """Async counterpart to :func:`parse_tool_events`."""
    state = _ParserState()
    async for event in events:
        for token in state.handle(event):
            yield token
