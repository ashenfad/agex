"""Convert a stream of provider-agnostic :class:`ToolCallEvent`\\ s into
:class:`TokenChunk`\\ s the agex :class:`ResponseBuilder` already knows
how to consume.

Two emission cadences:

* **Action tools** (``python_action`` / ``terminal_action``) stream
  through in real time — each JSON string key (``title``, ``thinking``,
  ``report``, ``code``/``commands``) emits content ``TokenChunk``\\ s
  as characters arrive and a closing ``TokenChunk`` when the value
  finishes.  This matches the XML tokenizer's cadence so the UI can
  stream agent text live.

* **File tools** (``write_file`` / ``edit_file``) buffer the whole
  argument JSON until ``ToolCallEnd`` and then emit a single
  ``TokenChunk(type="file_action", action=<built FileAction/EditAction>,
  done=True)``.  The tool arguments are already structured — no reason
  to re-serialize into the XML tokenizer's streaming shape just so the
  ResponseBuilder can parse it back.  Buffering also avoids the
  "content-before-path" ordering hazard; file contents don't need to
  stream for a usable UI.
"""

import json
from typing import AsyncIterator, Iterator

from agex.agent.datatypes import EditAction, FileAction
from agex.llm.core import TokenChunk
from agex.llm.formats.json_stream import JsonStringExtractor

from .events import ToolCallArgDelta, ToolCallEnd, ToolCallEvent, ToolCallStart
from .schemas import (
    ACTION_TOOLS,
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
    TOOL_WRITE_FILE,
)

_PYTHON_KEY_MAP = {
    "title": "title",
    "thinking": "thinking",
    "report": "report",
    "code": "python",
}

_TERMINAL_KEY_MAP = {
    "title": "title",
    "thinking": "thinking",
    "report": "report",
    "commands": "terminal",
}


class _CallState:
    """Per-tool-call state: streaming extractor for action tools, raw
    buffer for file tools."""

    __slots__ = ("tool_name", "_extractor", "_key_map", "_raw_buf")

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self._extractor: JsonStringExtractor | None = None
        self._key_map: dict[str, str] | None = None
        self._raw_buf: list[str] | None = None
        if tool_name in ACTION_TOOLS:
            self._extractor = JsonStringExtractor()
            self._key_map = (
                _PYTHON_KEY_MAP if tool_name == TOOL_PYTHON else _TERMINAL_KEY_MAP
            )
        else:
            self._raw_buf = []

    def feed_args(self, chunk: str) -> Iterator[TokenChunk]:
        if self._extractor is not None and self._key_map is not None:
            for delta in self._extractor.feed(chunk):
                token_type = self._key_map.get(delta.key)
                if token_type is None:
                    continue
                yield TokenChunk(
                    type=token_type,
                    content=delta.content,
                    done=delta.done,
                )
        elif self._raw_buf is not None:
            self._raw_buf.append(chunk)

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
        action: FileAction | EditAction | None
        if self.tool_name == TOOL_WRITE_FILE:
            action = _build_write_file(args)
        elif self.tool_name == TOOL_EDIT_FILE:
            action = _build_edit_file(args)
        else:
            action = None
        if action is not None:
            yield TokenChunk(
                type="file_action",
                content="",
                done=True,
                action=action,
            )


def _build_write_file(args: dict) -> FileAction | None:
    path = args.get("path") or ""
    if not path:
        return None
    content = args.get("content") or ""
    mode = args.get("mode") or "write"
    if mode not in ("write", "append"):
        mode = "write"
    return FileAction(path=path, content=content, mode=mode)  # type: ignore[arg-type]


def _build_edit_file(args: dict) -> EditAction | None:
    path = args.get("path") or ""
    search = args.get("search")
    if not path or search is None:
        return None
    if "replace" in args:
        operation = "replace"
        content = args["replace"]
    elif "insert_after" in args:
        operation = "insert-after"
        content = args["insert_after"]
    elif "insert_before" in args:
        operation = "insert-before"
        content = args["insert_before"]
    else:
        return None
    return EditAction(
        path=path,
        search=search,
        content=content,
        operation=operation,  # type: ignore[arg-type]
        match_all=bool(args.get("match_all", False)),
    )


class _ParserState:
    """Tracks all open tool calls across an event stream."""

    def __init__(self) -> None:
        self._calls: dict[str, _CallState] = {}

    def handle(self, event: ToolCallEvent) -> Iterator[TokenChunk]:
        if isinstance(event, ToolCallStart):
            self._calls[event.call_id] = _CallState(event.tool_name)
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
