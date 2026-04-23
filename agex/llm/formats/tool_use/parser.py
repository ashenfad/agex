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

* **File tools** (``write_file`` / ``edit_file``) buffer the whole
  argument JSON until ``ToolCallEnd`` and then emit a single
  ``TokenChunk(type="emission", emission=<FileWriteEmission | FileEditEmission>,
  done=True)``.  The builder just slots the prebuilt emission in place.
  File contents don't need live streaming for a usable UI, and
  buffering avoids the content-before-path ordering hazard.
"""

import json
from typing import AsyncIterator, Iterator

from agex.agent.emissions import FileEditEmission, FileWriteEmission
from agex.llm.core import TokenChunk
from agex.llm.formats.json_stream import JsonStringExtractor

from .events import ToolCallArgDelta, ToolCallEnd, ToolCallEvent, ToolCallStart
from .schemas import (
    ACTION_TOOLS,
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
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


class _CallState:
    """Per-tool-call state: streaming extractor for action tools, raw
    buffer for file tools.

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
                    emission_index=self.emission_index,
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
    return FileEditEmission(
        path=path,
        search=search,
        content=content,
        operation=operation,  # type: ignore[arg-type]
        match_all=bool(args.get("match_all", False)),
    )


class _ParserState:
    """Tracks open tool calls and assigns monotonic emission indices."""

    def __init__(self) -> None:
        self._calls: dict[str, _CallState] = {}
        self._next_index: int = 0

    def handle(self, event: ToolCallEvent) -> Iterator[TokenChunk]:
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
