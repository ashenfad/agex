"""Server-Sent Events (SSE) line parser.

Parses a stream of raw bytes/text from an HTTP response into SSE data lines.
Used by pyfetch-based LLM clients that don't have SDK-level SSE handling.
"""

from typing import AsyncIterator


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
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    return
                yield payload
            # Skip empty lines, comments (: ...), and other SSE fields
