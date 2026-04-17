"""Server-Sent Events (SSE) line parser.

Parses a stream of raw bytes/text from an HTTP response into SSE data lines.
Used by pyfetch-based LLM clients that don't have SDK-level SSE handling.
"""

from typing import AsyncIterator

_MAX_LINE_LENGTH = 1_048_576  # 1 MB guard against malformed streams


async def parse_sse_events(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Parse SSE text stream into data payloads.

    Handles the SSE wire format:
        data: {"chunk": "..."}
        data: [DONE]

    Args:
        chunks: Async iterator of raw text chunks from the HTTP response.

    Yields:
        The content of each ``data:`` line (stripped), excluding ``[DONE]``.
    """
    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        if "\n" not in buffer and len(buffer) > _MAX_LINE_LENGTH:
            raise ValueError(
                f"SSE line exceeded {_MAX_LINE_LENGTH} bytes without a newline"
            )
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    return
                yield payload
            # Skip empty lines, comments (: ...), and other SSE fields

    # Flush any trailing data not terminated by a newline (e.g. the
    # server closed the connection before sending a final newline).
    line = buffer.rstrip("\r")
    if line.startswith("data: "):
        payload = line[6:]
        if payload != "[DONE]":
            yield payload
