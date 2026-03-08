"""Tests for SSE line parser."""

import pytest

from agex.llm.sse import parse_sse_events


async def _collect(chunks):
    """Helper to collect async iterator results."""
    results = []
    async for item in chunks:
        results.append(item)
    return results


async def _aiter(*strings):
    """Create an async iterator from strings."""
    for s in strings:
        yield s


@pytest.mark.asyncio
async def test_basic_data_lines():
    raw = _aiter('data: {"a":1}\n\ndata: {"b":2}\n\n')
    results = await _collect(parse_sse_events(raw))
    assert results == ['{"a":1}', '{"b":2}']


@pytest.mark.asyncio
async def test_done_sentinel_stops():
    raw = _aiter('data: {"a":1}\n\ndata: [DONE]\n\ndata: {"b":2}\n\n')
    results = await _collect(parse_sse_events(raw))
    assert results == ['{"a":1}']


@pytest.mark.asyncio
async def test_chunked_across_boundaries():
    """SSE data split across multiple chunks."""
    raw = _aiter("dat", 'a: {"x":', '"y"}\n\n', 'data: {"z":1}\n\n')
    results = await _collect(parse_sse_events(raw))
    assert results == ['{"x":"y"}', '{"z":1}']


@pytest.mark.asyncio
async def test_ignores_comments_and_empty_lines():
    raw = _aiter(": comment\n\ndata: hello\n\n\n")
    results = await _collect(parse_sse_events(raw))
    assert results == ["hello"]


@pytest.mark.asyncio
async def test_carriage_return_handling():
    """Lines with \\r\\n endings."""
    raw = _aiter("data: one\r\n\r\ndata: two\r\n\r\n")
    results = await _collect(parse_sse_events(raw))
    assert results == ["one", "two"]


@pytest.mark.asyncio
async def test_empty_stream():
    raw = _aiter("")
    results = await _collect(parse_sse_events(raw))
    assert results == []


@pytest.mark.asyncio
async def test_multiple_chunks_per_line():
    """A single SSE line split across many tiny chunks."""
    raw = _aiter("d", "a", "t", "a", ":", " ", "ok", "\n", "\n")
    results = await _collect(parse_sse_events(raw))
    assert results == ["ok"]
