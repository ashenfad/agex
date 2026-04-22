"""Tests for the streaming JSON string-value extractor."""

import pytest

from agex.llm.formats.json_stream import (
    JsonStringDelta,
    JsonStringExtractor,
    iter_json_strings,
)


def _accumulate(chunks):
    """Feed chunks and collect (key, content) pairs.

    Returns a dict mapping key -> accumulated string, using the close
    deltas as completion markers.  Useful for equivalence assertions.
    """
    result: dict[str, str] = {}
    pending: dict[str, list[str]] = {}
    deltas = list(iter_json_strings(iter(chunks)))
    for d in deltas:
        if d.done:
            result[d.key] = "".join(pending.get(d.key, []))
        else:
            pending.setdefault(d.key, []).append(d.content)
    return result, deltas


class TestBasic:
    def test_single_string_value(self):
        result, deltas = _accumulate(['{"a": "hello"}'])
        assert result == {"a": "hello"}
        # Final delta for the value should be a done marker.
        close_deltas = [d for d in deltas if d.done]
        assert close_deltas == [JsonStringDelta("a", "", True)]

    def test_empty_string_value(self):
        result, deltas = _accumulate(['{"a": ""}'])
        assert result == {"a": ""}
        # Even empty strings should emit a done delta.
        assert any(d.key == "a" and d.done for d in deltas)

    def test_multiple_keys(self):
        result, _ = _accumulate(['{"a": "one", "b": "two", "c": "three"}'])
        assert result == {"a": "one", "b": "two", "c": "three"}

    def test_whitespace_between_tokens(self):
        result, _ = _accumulate(['{  "a"  :  "x"  ,  "b"  :  "y"  }'])
        assert result == {"a": "x", "b": "y"}


class TestNonStringValues:
    def test_number_value_skipped(self):
        result, deltas = _accumulate(['{"a": 42, "b": "hi"}'])
        assert result == {"b": "hi"}
        # No deltas for "a" at all.
        assert all(d.key != "a" for d in deltas)

    def test_bool_and_null_skipped(self):
        result, _ = _accumulate(['{"a": true, "b": false, "c": null, "d": "hi"}'])
        assert result == {"d": "hi"}

    def test_nested_array_skipped(self):
        result, _ = _accumulate(['{"a": [1, 2, "inside"], "b": "outside"}'])
        assert result == {"b": "outside"}

    def test_nested_object_skipped(self):
        result, _ = _accumulate(['{"a": {"x": "inside"}, "b": "outside"}'])
        assert result == {"b": "outside"}

    def test_nested_object_with_braces_and_commas_in_strings(self):
        result, _ = _accumulate(
            ['{"a": {"x": "has}brace, and comma"}, "b": "outside"}']
        )
        assert result == {"b": "outside"}

    def test_deeply_nested_skip(self):
        result, _ = _accumulate(['{"a": [{"x": [1, {"y": "z"}]}], "b": "final"}'])
        assert result == {"b": "final"}


class TestEscapes:
    def test_simple_escapes(self):
        result, _ = _accumulate([r'{"a": "line1\nline2\ttab\"quote\\back"}'])
        assert result == {"a": 'line1\nline2\ttab"quote\\back'}

    def test_forward_slash_escape(self):
        result, _ = _accumulate([r'{"a": "path\/to\/file"}'])
        assert result == {"a": "path/to/file"}

    def test_backspace_and_formfeed(self):
        result, _ = _accumulate([r'{"a": "x\by\fz"}'])
        assert result == {"a": "x\by\fz"}

    def test_unicode_escape_bmp(self):
        result, _ = _accumulate([r'{"a": "caf\u00e9"}'])
        assert result == {"a": "café"}

    def test_unicode_escape_ascii(self):
        result, _ = _accumulate([r'{"a": "\u0041\u0042"}'])
        assert result == {"a": "AB"}


class TestChunkSplits:
    def test_split_between_chars(self):
        result, _ = _accumulate(['{"a": "hel', 'lo"}'])
        assert result == {"a": "hello"}

    def test_split_across_key_name(self):
        result, _ = _accumulate(['{"ab', 'c": "x"}'])
        assert result == {"abc": "x"}

    def test_split_mid_escape(self):
        # Chunk boundary between backslash and escape char.
        result, _ = _accumulate(['{"a": "line1\\', 'nline2"}'])
        assert result == {"a": "line1\nline2"}

    def test_split_mid_unicode_hex(self):
        # Split \u00e9 across multiple chunks at various points.
        cases = [
            [r'{"a": "\u', r'00e9"}'],
            [r'{"a": "\u0', r'0e9"}'],
            [r'{"a": "\u00', r'e9"}'],
            [r'{"a": "\u00e', r'9"}'],
        ]
        for chunks in cases:
            result, _ = _accumulate(chunks)
            assert result == {"a": "é"}, f"failed for split {chunks}"

    def test_char_by_char_feed(self):
        # Degenerate case: one character per chunk.
        s = '{"title": "hi", "count": 3, "msg": "ok"}'
        chunks = list(s)
        result, _ = _accumulate(chunks)
        assert result == {"title": "hi", "msg": "ok"}

    def test_split_at_colon(self):
        result, _ = _accumulate(['{"a"', ":", ' "x"}'])
        assert result == {"a": "x"}

    def test_split_at_closing_quote(self):
        result, deltas = _accumulate(['{"a": "hi', '"}'])
        assert result == {"a": "hi"}
        # The close delta should come after the second chunk, not the first.
        close_idx = next(i for i, d in enumerate(deltas) if d.done)
        content_deltas_before_close = [d for d in deltas[:close_idx] if not d.done]
        assert "".join(d.content for d in content_deltas_before_close) == "hi"


class TestStreamingCadence:
    def test_content_flushed_per_chunk(self):
        """Each chunk with string content should emit at least one delta."""
        deltas = list(iter_json_strings(iter(['{"a": "foo', "bar", 'baz"}'])))
        # Collect content deltas for "a" (ignore done).
        content_deltas = [d for d in deltas if d.key == "a" and not d.done]
        # We got chunks across three feeds — expect ~3 content deltas.
        assert len(content_deltas) >= 2
        assert "".join(d.content for d in content_deltas) == "foobarbaz"

    def test_done_fires_before_next_key_content(self):
        """Close delta for key A must fire before content delta for key B."""
        deltas = list(iter_json_strings(iter(['{"a": "x", "b": "y"}'])))
        # Find index of done for "a" and any content for "b".
        a_done_idx = next(i for i, d in enumerate(deltas) if d.key == "a" and d.done)
        b_any_idx = next(i for i, d in enumerate(deltas) if d.key == "b")
        assert a_done_idx < b_any_idx


class TestExtractorInstance:
    def test_multiple_feeds_share_state(self):
        ext = JsonStringExtractor()
        deltas: list[JsonStringDelta] = []
        deltas.extend(ext.feed('{"a": "hello, '))
        deltas.extend(ext.feed('world"}'))
        # Reconstruct.
        acc: dict[str, str] = {}
        pending: dict[str, list[str]] = {}
        for d in deltas:
            if d.done:
                acc[d.key] = "".join(pending.get(d.key, []))
            else:
                pending.setdefault(d.key, []).append(d.content)
        assert acc == {"a": "hello, world"}

    def test_empty_object(self):
        result, deltas = _accumulate(["{}"])
        assert result == {}
        assert deltas == []

    def test_empty_input(self):
        result, deltas = _accumulate([""])
        assert result == {}
        assert deltas == []


class TestAsync:
    @pytest.mark.asyncio
    async def test_aiter_basic(self):
        from agex.llm.formats.json_stream import aiter_json_strings

        async def chunks():
            yield '{"a": "hel'
            yield 'lo", "b":'
            yield ' "world"}'

        acc: dict[str, str] = {}
        pending: dict[str, list[str]] = {}
        async for d in aiter_json_strings(chunks()):
            if d.done:
                acc[d.key] = "".join(pending.get(d.key, []))
            else:
                pending.setdefault(d.key, []).append(d.content)
        assert acc == {"a": "hello", "b": "world"}
