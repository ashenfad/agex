"""
XML utilities for LLM streaming support.

Provides parsing utilities and data types for XML-formatted LLM responses.
All utilities are optional - clients can use these or implement custom logic.

Note: For rendering events to XML, see agex.render.xml.render_events_as_xml()
"""

import re
from dataclasses import dataclass
from typing import Iterator

from agex.llm.core import ResponseParseError

# XML tag names as constants
TAG_THINKING = "THINKING"
TAG_PYTHON = "PYTHON"
TAG_TITLE = "TITLE"  # Optional, for future use


@dataclass
class XMLResponse:
    """Parsed XML response from LLM."""

    thinking: str
    code: str
    title: str = ""  # Optional for now, will be required in Phase 2.5


# System prompt instructions for XML format
XML_FORMAT_PRIMER = f"""
Format your response using XML tags:

<{TAG_THINKING}>Your step-by-step reasoning here</{TAG_THINKING}>
<{TAG_PYTHON}>
# Your Python code here
</{TAG_PYTHON}>

Example:
<{TAG_THINKING}>
I need to calculate the sum of the numbers and return it.
</{TAG_THINKING}>
<{TAG_PYTHON}>
total = sum(numbers)
task_success(total)
</{TAG_PYTHON}>
"""


def parse_xml_response(xml_text: str) -> XMLResponse:
    """
    Parse complete XML response (non-streaming).

    Extracts <THINKING> and <PYTHON> tags from complete text.
    Tags are case-insensitive.

    Args:
        xml_text: Complete XML response text

    Returns:
        XMLResponse with thinking and code fields

    Raises:
        ResponseParseError: If required tags are missing or malformed
    """
    # Extract thinking (case-insensitive)
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

    # Extract code (case-insensitive)
    code_match = re.search(
        rf"<{TAG_PYTHON}>(.*?)</{TAG_PYTHON}>", xml_text, re.DOTALL | re.IGNORECASE
    )
    if not code_match:
        raise ResponseParseError(
            f"Missing <{TAG_PYTHON}> tags in XML response. "
            f"Response: {xml_text[:200]}..."
        )
    code = code_match.group(1).strip()

    # Extract optional title (case-insensitive)
    title = ""
    title_match = re.search(
        rf"<{TAG_TITLE}>(.*?)</{TAG_TITLE}>", xml_text, re.DOTALL | re.IGNORECASE
    )
    if title_match:
        title = title_match.group(1).strip()

    return XMLResponse(thinking=thinking, code=code, title=title)


@dataclass
class TokenChunk:
    """
    A piece of streamed content from the LLM.

    Not an Event - tokens are ephemeral and don't go in the state log.

    Attributes:
        type: Either "thinking" or "python" (future: "title")
        content: The text content (incremental)
        done: True when this section is complete
    """

    type: str  # Literal["thinking", "python"] but using str for now
    content: str
    done: bool = False


def _process_section_closing(
    buffer: str,
    section_type: str,
    closing_tag: str,
) -> tuple[list[TokenChunk], str, bool]:
    """
    Helper to process closing tag for a section.

    Args:
        buffer: Current buffer content
        section_type: Type of section ("thinking" or "python")
        closing_tag: Closing tag to search for (e.g., TAG_THINKING)

    Returns:
        Tuple of (tokens_to_yield, updated_buffer, section_complete)
        - tokens_to_yield: List of TokenChunk objects to yield
        - updated_buffer: Buffer with processed content removed
        - section_complete: True if closing tag was found, False otherwise
    """
    closing = re.search(rf"</{closing_tag}>", buffer, re.IGNORECASE)
    if closing:
        # Found closing tag - yield all content before it
        tokens = []
        before_tag = buffer[: closing.start()]
        if before_tag:
            tokens.append(TokenChunk(type=section_type, content=before_tag, done=False))
        tokens.append(TokenChunk(type=section_type, content="", done=True))

        # Keep content after closing tag
        updated_buffer = buffer[closing.end() :]
        return tokens, updated_buffer, True
    else:
        # No closing tag yet - yield content if substantial
        # Yield on: length threshold OR any whitespace (natural word boundaries)
        tokens = []
        if len(buffer) > 10 or any(c.isspace() for c in buffer):
            tokens.append(TokenChunk(type=section_type, content=buffer, done=False))
            updated_buffer = ""
        else:
            updated_buffer = buffer
        return tokens, updated_buffer, False


def tokenize_xml_stream(raw_chunks: Iterator[str]) -> Iterator[TokenChunk]:
    """
    Convert raw text stream to TokenChunks via XML parsing.

    This is a shared utility that handles buffering and tag detection.
    Clients can use this or implement their own tokenization logic.

    Architecture:
        Provider raw stream → Iterator[str] → tokenize_xml_stream → Iterator[TokenChunk]

    Args:
        raw_chunks: Iterator of raw text chunks from provider

    Yields:
        TokenChunk objects as sections are parsed

    Raises:
        ResponseParseError: If XML structure is malformed
    """
    buffer = ""
    current_section = None  # "thinking" | "python" | None

    for chunk in raw_chunks:
        buffer += chunk

        # Process all complete tags in the buffer
        while True:
            if current_section is None:
                # Look for opening tags (case-insensitive)
                thinking_start = re.search(rf"<{TAG_THINKING}>", buffer, re.IGNORECASE)
                python_start = re.search(rf"<{TAG_PYTHON}>", buffer, re.IGNORECASE)

                if thinking_start:
                    # Found opening thinking tag
                    current_section = "thinking"
                    # Keep any content after the tag in buffer
                    buffer = buffer[thinking_start.end() :]
                    continue
                elif python_start:
                    # Found opening python tag
                    current_section = "python"
                    # Keep any content after the tag in buffer
                    buffer = buffer[python_start.end() :]
                    continue
                else:
                    # No opening tag found yet
                    break

            # We're in a section - look for closing tag
            if current_section == "thinking":
                tokens, buffer, complete = _process_section_closing(
                    buffer, "thinking", TAG_THINKING
                )
                for token in tokens:
                    yield token
                if complete:
                    current_section = None
                    continue
                else:
                    break

            elif current_section == "python":
                tokens, buffer, complete = _process_section_closing(
                    buffer, "python", TAG_PYTHON
                )
                for token in tokens:
                    yield token
                if complete:
                    current_section = None
                    continue
                else:
                    break

    # Handle any remaining buffer at end of stream
    if buffer and current_section:
        # Still in a section but stream ended - yield remaining content
        yield TokenChunk(type=current_section, content=buffer, done=False)
        # Note: We don't mark done=True here because the section wasn't properly closed
        # This could indicate a truncated response
