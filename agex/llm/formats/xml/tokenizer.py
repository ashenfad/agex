"""Streaming XML tokenizer.

Consumes a text-chunk stream from the provider and yields
:class:`TokenChunk`\\ s as sections are recognised. Tolerates implicit
closes (sibling tag opener at a line start while the previous section
is still open) and emits trailing deltas conservatively so we don't
break a ``</TAG>`` sequence across a chunk boundary.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Literal

from agex.llm.core import ResponseParseError, TokenChunk

from .tags import (
    TAG_EDIT,
    TAG_FILE,
    TAG_INSERT_AFTER,
    TAG_INSERT_BEFORE,
    TAG_PYTHON,
    TAG_REPLACE,
    TAG_REPORT,
    TAG_SEARCH,
    TAG_TERMINAL,
    TAG_THINKING,
    TAG_TITLE,
)
from .validation import (
    validate_edit_search,
    validate_file_mode,
    validate_file_path,
)

if TYPE_CHECKING:
    from agex.agent.datatypes import EditAction, FileAction


# Sibling opener pattern for implicit-close recovery. If any of these
# top-level tags appears at the start of a line (optionally indented)
# while we're inside another section, we assume the agent forgot to
# close the current section and transition to the new one. Tags with
# attributes (FILE, EDIT) require whitespace after the tag name;
# simple tags require the closing ">" immediately.
_IMPLICIT_CLOSE_PATTERN = re.compile(
    rf"(?:^|\n)[\t ]*<(?:"
    rf"{TAG_TITLE}>|"
    rf"{TAG_THINKING}>|"
    rf"{TAG_REPORT}>|"
    rf"{TAG_PYTHON}>|"
    rf"{TAG_TERMINAL}>|"
    rf"{TAG_FILE}\s|"
    rf"{TAG_EDIT}\s"
    rf")",
    re.IGNORECASE,
)


def _find_implicit_close_pos(buffer: str, current_section: str) -> int | None:
    """Find the position of a sibling top-level opener (the ``<``
    character) that should implicitly close the current section.

    Returns None if no such boundary is found. The opener tag remains
    in the buffer so the tokenizer can start the new section on its
    next pass.
    """
    match = _IMPLICIT_CLOSE_PATTERN.search(buffer)
    if match is None:
        return None
    return buffer.find("<", match.start(), match.end())


@dataclass
class XMLResponse:
    """Parsed XML response (non-streaming; used by ``parse_xml_response``)."""

    thinking: str
    code: str
    file_actions: list["FileAction | EditAction"] = field(default_factory=list)
    terminal: str | None = None
    title: str = ""
    report: str = ""


def parse_xml_response(xml_text: str) -> XMLResponse:
    """Parse a complete (non-streaming) XML response text.

    Extracts ``<TITLE>``, ``<THINKING>``, ``<FILE>``, ``<EDIT>``, and
    ``<PYTHON>`` or ``<TERMINAL>`` tags. Tags are case-insensitive.
    Requires exactly one of ``<PYTHON>`` / ``<TERMINAL>``.

    Raises :class:`ResponseParseError` if required tags are missing or
    malformed.
    """
    thinking_match = re.search(
        rf"<{TAG_THINKING}>(.*?)</{TAG_THINKING}>",
        xml_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not thinking_match:
        raise ResponseParseError(
            f"Missing <{TAG_THINKING}> tags in XML response. "
            f"Response: {xml_text[:200]}..."
        )
    thinking = thinking_match.group(1).strip()

    terminal_match = re.search(
        rf"<{TAG_TERMINAL}>(.*?)</{TAG_TERMINAL}>",
        xml_text,
        re.DOTALL | re.IGNORECASE,
    )
    terminal = terminal_match.group(1).strip() if terminal_match else None

    code_match = re.search(
        rf"<{TAG_PYTHON}>(.*?)</{TAG_PYTHON}>", xml_text, re.DOTALL | re.IGNORECASE
    )

    if terminal and code_match:
        raise ResponseParseError(
            f"Response contains both <{TAG_TERMINAL}> and <{TAG_PYTHON}>. "
            f"Use one or the other."
        )
    if not terminal and not code_match:
        raise ResponseParseError(
            f"Missing <{TAG_TERMINAL}> or <{TAG_PYTHON}> tags in XML response. "
            f"Response: {xml_text[:200]}..."
        )

    code = code_match.group(1).strip() if code_match else ""

    title = ""
    title_match = re.search(
        rf"<{TAG_TITLE}>(.*?)</{TAG_TITLE}>", xml_text, re.DOTALL | re.IGNORECASE
    )
    if title_match:
        title = title_match.group(1).strip()

    report = ""
    report_match = re.search(
        rf"<{TAG_REPORT}>(.*?)</{TAG_REPORT}>", xml_text, re.DOTALL | re.IGNORECASE
    )
    if report_match:
        report = report_match.group(1).strip()

    from agex.agent.datatypes import EditAction, FileAction

    all_matches: list[tuple[int, "FileAction | EditAction"]] = []

    file_matches = re.finditer(
        rf"<{TAG_FILE}\s+([^>]*?)>(.*?)</{TAG_FILE}>",
        xml_text,
        re.DOTALL | re.IGNORECASE,
    )
    for match in file_matches:
        attrs_text = match.group(1)
        content = match.group(2)

        path_match = re.search(r'path=["\'](.*?)["\']', attrs_text, re.IGNORECASE)
        mode_match = re.search(r'mode=["\'](.*?)["\']', attrs_text, re.IGNORECASE)

        if path_match:
            path = validate_file_path(path_match.group(1))
            mode_str = mode_match.group(1).strip() if mode_match else "write"
            mode = validate_file_mode(mode_str, path)
            all_matches.append(
                (match.start(), FileAction(path=path, content=content, mode=mode))
            )

    edit_matches = re.finditer(
        rf"<{TAG_EDIT}\s+([^>]*?)>(.*?)</{TAG_EDIT}>",
        xml_text,
        re.DOTALL | re.IGNORECASE,
    )
    for match in edit_matches:
        attrs_text = match.group(1)
        inner_content = match.group(2)

        path_match = re.search(r'path=["\'](.*?)["\']', attrs_text, re.IGNORECASE)
        match_all_match = re.search(
            r'match_all=["\'](.*?)["\']', attrs_text, re.IGNORECASE
        )

        if path_match:
            path = validate_file_path(path_match.group(1))
            match_all = (
                match_all_match is not None
                and match_all_match.group(1).lower() == "true"
            )

            search_match = re.search(
                rf"<{TAG_SEARCH}>(.*?)</{TAG_SEARCH}>",
                inner_content,
                re.DOTALL | re.IGNORECASE,
            )
            replace_match = re.search(
                rf"<{TAG_REPLACE}>(.*?)</{TAG_REPLACE}>",
                inner_content,
                re.DOTALL | re.IGNORECASE,
            )
            insert_after_match = re.search(
                rf"<{TAG_INSERT_AFTER}>(.*?)</{TAG_INSERT_AFTER}>",
                inner_content,
                re.DOTALL | re.IGNORECASE,
            )
            insert_before_match = re.search(
                rf"<{TAG_INSERT_BEFORE}>(.*?)</{TAG_INSERT_BEFORE}>",
                inner_content,
                re.DOTALL | re.IGNORECASE,
            )

            if not search_match:
                raise ResponseParseError(
                    f'<EDIT path="{path}"> is missing <SEARCH> tag. '
                    f"EDIT requires <SEARCH> and one of <REPLACE>, <INSERT-AFTER>, or <INSERT-BEFORE>. "
                    f'To append content to a file, use <FILE mode="append"> instead.'
                )

            operation_matches = [
                (m, op)
                for m, op in [
                    (replace_match, "replace"),
                    (insert_after_match, "insert-after"),
                    (insert_before_match, "insert-before"),
                ]
                if m
            ]

            if len(operation_matches) == 0:
                raise ResponseParseError(
                    f'<EDIT path="{path}"> is missing an operation tag. '
                    f"EDIT requires one of <REPLACE>, <INSERT-AFTER>, or <INSERT-BEFORE>."
                )
            elif len(operation_matches) > 1:
                raise ResponseParseError(
                    f'<EDIT path="{path}"> contains multiple operation tags. '
                    f"Only one of <REPLACE>, <INSERT-AFTER>, or <INSERT-BEFORE> is allowed per <EDIT> block."
                )

            op_match, operation = operation_matches[0]
            content = op_match.group(1)

            search = search_match.group(1)

            validate_edit_search(path, search)
            all_matches.append(
                (
                    match.start(),
                    EditAction(
                        path=path,
                        search=search,
                        content=content,
                        operation=operation,
                        match_all=match_all,
                    ),
                )
            )

    all_matches.sort(key=lambda x: x[0])
    file_actions = [action for _, action in all_matches]

    return XMLResponse(
        thinking=thinking,
        code=code,
        terminal=terminal,
        title=title,
        report=report,
        file_actions=file_actions,
    )


class _XMLTokenizerState:
    """Shared state machine for XML tokenization.

    Encapsulates the buffer and section tracking logic used by both
    sync and async tokenizers.
    """

    _CLOSING_TAGS = {
        "title": TAG_TITLE,
        "thinking": TAG_THINKING,
        "report": TAG_REPORT,
        "python": TAG_PYTHON,
        "terminal": TAG_TERMINAL,
        "file": TAG_FILE,
        "edit": TAG_EDIT,
    }

    def __init__(self) -> None:
        self.buffer = ""
        self.current_section: (
            Literal["title", "thinking", "report", "python", "terminal", "file", "edit"]
            | None
        ) = None

    def add_chunk(self, chunk: str) -> None:
        self.buffer += chunk

    def process_buffer(self) -> tuple[list[TokenChunk], bool]:
        """Process buffer and return tokens + should-stop flag.

        ``should_stop`` is True after ``</PYTHON>`` or ``</TERMINAL>``.
        """
        tokens: list[TokenChunk] = []
        should_stop = False

        while True:
            if self.current_section is None:
                section_tokens, found = self._try_start_section()
                tokens.extend(section_tokens)
                if not found:
                    break
            else:
                section_tokens, complete, stop = self._process_current_section()
                tokens.extend(section_tokens)
                if stop:
                    should_stop = True
                    break
                if not complete:
                    break

        return tokens, should_stop

    def finalize(self) -> list[TokenChunk]:
        """Emit any remaining buffered content at end of stream."""
        if self.buffer and self.current_section:
            return [
                TokenChunk(type=self.current_section, content=self.buffer, done=False)
            ]
        return []

    def _try_start_section(self) -> tuple[list[TokenChunk], bool]:
        title_start = re.search(rf"<{TAG_TITLE}>", self.buffer, re.IGNORECASE)
        thinking_start = re.search(rf"<{TAG_THINKING}>", self.buffer, re.IGNORECASE)
        report_start = re.search(rf"<{TAG_REPORT}>", self.buffer, re.IGNORECASE)
        python_start = re.search(rf"<{TAG_PYTHON}>", self.buffer, re.IGNORECASE)
        terminal_start = re.search(rf"<{TAG_TERMINAL}>", self.buffer, re.IGNORECASE)
        file_start = re.search(rf"<{TAG_FILE}\s+([^>]*?)>", self.buffer, re.IGNORECASE)
        edit_start = re.search(rf"<{TAG_EDIT}\s+([^>]*?)>", self.buffer, re.IGNORECASE)

        starts: list[tuple[int, str, int, str | None]] = []
        if title_start:
            starts.append((title_start.start(), "title", title_start.end(), None))
        if thinking_start:
            starts.append(
                (thinking_start.start(), "thinking", thinking_start.end(), None)
            )
        if report_start:
            starts.append((report_start.start(), "report", report_start.end(), None))
        if python_start:
            starts.append((python_start.start(), "python", python_start.end(), None))
        if terminal_start:
            starts.append(
                (terminal_start.start(), "terminal", terminal_start.end(), None)
            )
        if file_start:
            attrs_text = file_start.group(1)
            path_match = re.search(r'path=["\'](.*?)["\']', attrs_text, re.IGNORECASE)
            mode_match = re.search(r'mode=["\'](.*?)["\']', attrs_text, re.IGNORECASE)

            if path_match:
                path = validate_file_path(path_match.group(1))
                mode_str = mode_match.group(1).strip() if mode_match else "write"
                mode = validate_file_mode(mode_str, path)
                starts.append(
                    (
                        file_start.start(),
                        "file",
                        file_start.end(),
                        f"path={path},mode={mode}",
                    )
                )
        if edit_start:
            attrs_text = edit_start.group(1)
            path_match = re.search(r'path=["\'](.*?)["\']', attrs_text, re.IGNORECASE)
            match_all_match = re.search(
                r'match_all=["\'](.*?)["\']', attrs_text, re.IGNORECASE
            )

            if path_match:
                path = validate_file_path(path_match.group(1))
                match_all = (
                    match_all_match is not None
                    and match_all_match.group(1).lower() == "true"
                )
                starts.append(
                    (
                        edit_start.start(),
                        "edit",
                        edit_start.end(),
                        f"path={path},match_all={match_all}",
                    )
                )

        if not starts:
            return [], False

        starts.sort()
        _, section, end_pos, metadata = starts[0]

        self.current_section = section  # type: ignore[assignment]
        self.buffer = self.buffer[end_pos:]

        tokens: list[TokenChunk] = []
        if section == "file" and metadata:
            tokens.append(TokenChunk(type="file", content=metadata, done=False))
        elif section == "edit" and metadata:
            tokens.append(TokenChunk(type="edit", content=metadata, done=False))

        return tokens, True

    def _process_current_section(self) -> tuple[list[TokenChunk], bool, bool]:
        assert self.current_section is not None

        closing_tag = self._CLOSING_TAGS[self.current_section]
        tokens, new_buffer, complete = self._process_section_closing(
            self.buffer, self.current_section, closing_tag
        )
        self.buffer = new_buffer

        should_stop = False
        if complete:
            if self.current_section in ("python", "terminal"):
                # Enforce single turn: stop after first Python or Terminal section.
                should_stop = True
            self.current_section = None

        return tokens, complete, should_stop

    @staticmethod
    def _process_section_closing(
        buffer: str,
        section_type: Literal[
            "title", "thinking", "report", "python", "terminal", "file", "edit"
        ],
        closing_tag: str,
    ) -> tuple[list[TokenChunk], str, bool]:
        closing = re.search(rf"</{closing_tag}>", buffer, re.IGNORECASE)
        closing_pos = closing.start() if closing else None

        # Check for implicit-close boundary (sibling top-level tag opener
        # at a line start). If it appears before the proper closing tag,
        # assume the agent forgot to close and transition early.
        implicit_pos = _find_implicit_close_pos(buffer, section_type)
        if implicit_pos is not None and (
            closing_pos is None or implicit_pos < closing_pos
        ):
            tokens = []
            before = buffer[:implicit_pos].rstrip("\n\t ")
            if before:
                tokens.append(TokenChunk(type=section_type, content=before, done=False))
            tokens.append(TokenChunk(type=section_type, content="", done=True))
            # Keep the sibling opener in the buffer so the tokenizer can
            # pick it up on the next pass.
            return tokens, buffer[implicit_pos:], True

        if closing:
            tokens = []
            before_tag = buffer[: closing.start()]
            if before_tag:
                tokens.append(
                    TokenChunk(type=section_type, content=before_tag, done=False)
                )
            tokens.append(TokenChunk(type=section_type, content="", done=True))

            updated_buffer = buffer[closing.end() :]
            return tokens, updated_buffer, True

        # No closing tag yet — yield content but hold back potential tag
        # starts so we don't split "</TAG>" across chunks.
        tokens = []
        last_bracket = buffer.rfind("<")

        if last_bracket == -1:
            if len(buffer) > 10 or any(c.isspace() for c in buffer):
                tokens.append(TokenChunk(type=section_type, content=buffer, done=False))
                updated_buffer = ""
            else:
                updated_buffer = buffer
        else:
            content_to_yield = buffer[:last_bracket]
            holdback = buffer[last_bracket:]

            if content_to_yield and (
                len(content_to_yield) > 10 or any(c.isspace() for c in content_to_yield)
            ):
                tokens.append(
                    TokenChunk(type=section_type, content=content_to_yield, done=False)
                )
                updated_buffer = holdback
            else:
                updated_buffer = buffer

        return tokens, updated_buffer, False


def tokenize_xml_stream(raw_chunks: Iterator[str]) -> Iterator[TokenChunk]:
    """Convert a raw text stream to ``TokenChunk``\\ s via XML parsing.

    Architecture:
        provider raw stream → Iterator[str] → tokenize_xml_stream → Iterator[TokenChunk]
    """
    state = _XMLTokenizerState()
    for chunk in raw_chunks:
        state.add_chunk(chunk)
        tokens, should_stop = state.process_buffer()
        yield from tokens
        if should_stop:
            return
    yield from state.finalize()


async def atokenize_xml_stream(
    raw_chunks: AsyncIterator[str],
) -> AsyncIterator[TokenChunk]:
    """Async counterpart to :func:`tokenize_xml_stream`."""
    state = _XMLTokenizerState()
    async for chunk in raw_chunks:
        state.add_chunk(chunk)
        tokens, should_stop = state.process_buffer()
        for token in tokens:
            yield token
        if should_stop:
            return
    for token in state.finalize():
        yield token
