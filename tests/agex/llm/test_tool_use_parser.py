"""Tests for the tool-use parser (ToolCallEvent stream → TokenChunks)."""

import json

import pytest

from agex.agent.emissions import FileEditEmission, FileWriteEmission
from agex.llm.core import TokenChunk
from agex.llm.formats.tool_use import (
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
    TOOL_TERMINAL,
    TOOL_WRITE_FILE,
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallStart,
    parse_tool_events,
)


def _tokens(events):
    return list(parse_tool_events(iter(events)))


def _chunked(call_id: str, s: str, n: int = 4):
    """Slice a JSON payload into n-byte ArgDelta events."""
    return [ToolCallArgDelta(call_id, s[i : i + n]) for i in range(0, len(s), n)]


class TestPythonAction:
    def test_basic(self):
        args = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})
        events = [
            ToolCallStart("c1", TOOL_PYTHON),
            *_chunked("c1", args, 6),
            ToolCallEnd("c1"),
        ]
        tokens = _tokens(events)

        # All tokens for the first call carry emission_index=0.
        assert all(t.emission_index == 0 for t in tokens)

        by_type: dict[str, list[TokenChunk]] = {}
        for t in tokens:
            by_type.setdefault(t.type, []).append(t)

        assert "".join(t.content for t in by_type["title"] if not t.done) == "t"
        assert "".join(t.content for t in by_type["thinking"] if not t.done) == "T"
        assert "".join(t.content for t in by_type["python"] if not t.done) == "print(1)"
        for type_name in ("title", "thinking", "python"):
            assert by_type[type_name][-1].done is True

    def test_report_maps_to_text_token(self):
        """The legacy ``report`` schema param streams as ``text`` tokens
        so the builder produces a TextEmission.
        """
        args = json.dumps(
            {
                "title": "t",
                "thinking": "T",
                "report": "Working...",
                "code": "pass",
            }
        )
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_PYTHON),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        text_content = "".join(
            t.content for t in tokens if t.type == "text" and not t.done
        )
        assert text_content == "Working..."

    def test_streaming_cadence_per_chunk(self):
        """Each chunk containing string content should emit at least one
        content delta, so the UI can render incrementally."""
        args = json.dumps({"title": "t", "thinking": "step by step", "code": "x"})
        chunks = _chunked("c1", args, 3)
        tokens = _tokens([ToolCallStart("c1", TOOL_PYTHON), *chunks, ToolCallEnd("c1")])
        thinking_content_deltas = [
            t for t in tokens if t.type == "thinking" and not t.done
        ]
        assert len(thinking_content_deltas) >= 2

    def test_tool_start_marker_emitted_for_python_action(self):
        """The parser emits a ``tool_start`` marker on every
        python_action so the builder knows the turn called a tool even
        if the content arg is empty."""
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_PYTHON),
                ToolCallArgDelta("c1", json.dumps({"code": ""})),
                ToolCallEnd("c1"),
            ]
        )
        starts = [t for t in tokens if t.type == "tool_start"]
        assert len(starts) == 1
        assert starts[0].content == "python_action"
        assert starts[0].emission_index == 0


class TestTerminalAction:
    def test_commands_field_maps_to_terminal_token(self):
        args = json.dumps(
            {"title": "t", "thinking": "T", "commands": "ls -la\necho hi"}
        )
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_TERMINAL),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        commands = "".join(
            t.content for t in tokens if t.type == "terminal" and not t.done
        )
        assert commands == "ls -la\necho hi"
        assert not any(t.type == "python" for t in tokens)


def _single_emission(tokens):
    """Assert exactly one ``emission`` TokenChunk and return its payload."""
    emission_tokens = [t for t in tokens if t.type == "emission"]
    assert len(emission_tokens) == 1
    token = emission_tokens[0]
    assert token.done is True
    assert token.content == ""
    assert token.emission is not None
    return token.emission


class TestWriteFile:
    def test_basic(self):
        args = json.dumps({"path": "/helpers/x.py", "content": "def f(): pass\n"})
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_WRITE_FILE),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        emission = _single_emission(tokens)
        assert isinstance(emission, FileWriteEmission)
        assert emission.path == "/helpers/x.py"
        assert emission.content == "def f(): pass\n"
        assert emission.mode == "write"

    def test_append_mode(self):
        args = json.dumps({"path": "/x.py", "content": "extra\n", "mode": "append"})
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_WRITE_FILE),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        emission = _single_emission(tokens)
        assert emission.mode == "append"

    def test_invalid_mode_coerces_to_write(self):
        args = json.dumps({"path": "/x.py", "content": "x", "mode": "garbage"})
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_WRITE_FILE),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        emission = _single_emission(tokens)
        assert emission.mode == "write"

    def test_batched_across_chunks(self):
        args = json.dumps({"path": "/a.py", "content": "x = 1"})
        events = [
            ToolCallStart("c1", TOOL_WRITE_FILE),
            *_chunked("c1", args, 5),
        ]
        # Before End: UI streaming tokens flow, but no authoritative
        # emission is built yet.
        pre_end = _tokens(events)
        assert pre_end  # streaming file_path / file_content tokens
        assert all(t.type != "emission" for t in pre_end)
        emission = _single_emission(_tokens([*events, ToolCallEnd("c1")]))
        assert emission.path == "/a.py"
        assert emission.content == "x = 1"

    def test_empty_content(self):
        args = json.dumps({"path": "/a.py", "content": ""})
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_WRITE_FILE),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        emission = _single_emission(tokens)
        assert emission.content == ""

    def test_missing_path_dropped(self):
        args = json.dumps({"content": "x = 1"})
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_WRITE_FILE),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        # UI may stream the content it sees, but nothing actionable is
        # finalized — no emission token.
        assert all(t.type != "emission" for t in tokens)

    def test_invalid_json_dropped(self):
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_WRITE_FILE),
                ToolCallArgDelta("c1", "not-json-at-all"),
                ToolCallEnd("c1"),
            ]
        )
        assert tokens == []

    def test_streams_path_and_content_as_args_arrive(self):
        """write_file args stream as file_path / file_content tokens
        so callers can watch file writes happen, not just the final
        buffered emission."""
        args = json.dumps({"path": "/a.py", "content": "x = 1\n"})
        events = [
            ToolCallStart("c1", TOOL_WRITE_FILE),
            *_chunked("c1", args, 3),
            ToolCallEnd("c1"),
        ]
        tokens = _tokens(events)
        path_content = "".join(
            t.content for t in tokens if t.type == "file_path" and not t.done
        )
        content_content = "".join(
            t.content for t in tokens if t.type == "file_content" and not t.done
        )
        assert path_content == "/a.py"
        assert content_content == "x = 1\n"
        # Authoritative emission still arrives at the end.
        assert _single_emission(tokens).path == "/a.py"


class TestEditFile:
    def test_replace(self):
        args = json.dumps(
            {
                "path": "/a.py",
                "search": "old_fn",
                "replace": "new_fn",
            }
        )
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_EDIT_FILE),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        emission = _single_emission(tokens)
        assert isinstance(emission, FileEditEmission)
        assert emission.path == "/a.py"
        assert emission.search == "old_fn"
        assert emission.content == "new_fn"
        assert emission.match_all is False

    def test_match_all_true(self):
        args = json.dumps(
            {
                "path": "/a.py",
                "search": "X",
                "replace": "Y",
                "match_all": True,
            }
        )
        emission = _single_emission(
            _tokens(
                [
                    ToolCallStart("c1", TOOL_EDIT_FILE),
                    ToolCallArgDelta("c1", args),
                    ToolCallEnd("c1"),
                ]
            )
        )
        assert emission.match_all is True

    def test_missing_replace_dropped(self):
        args = json.dumps({"path": "/a.py", "search": "X"})
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_EDIT_FILE),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        # UI sees the partial args, but without ``replace``, nothing
        # actionable is finalized.
        assert all(t.type != "emission" for t in tokens)

    def test_missing_search_dropped(self):
        args = json.dumps({"path": "/a.py", "replace": "Y"})
        tokens = _tokens(
            [
                ToolCallStart("c1", TOOL_EDIT_FILE),
                ToolCallArgDelta("c1", args),
                ToolCallEnd("c1"),
            ]
        )
        # Search is required for edits; UI can still stream what
        # arrived, but no emission is built.
        assert all(t.type != "emission" for t in tokens)

    def test_streams_search_and_replacement_as_args_arrive(self):
        """edit_file args stream as file_path / file_search /
        file_content tokens before the final emission."""
        args = json.dumps({"path": "/a.py", "search": "old", "replace": "new"})
        events = [
            ToolCallStart("c1", TOOL_EDIT_FILE),
            *_chunked("c1", args, 4),
            ToolCallEnd("c1"),
        ]
        tokens = _tokens(events)
        path = "".join(
            t.content for t in tokens if t.type == "file_path" and not t.done
        )
        search = "".join(
            t.content for t in tokens if t.type == "file_search" and not t.done
        )
        content = "".join(
            t.content for t in tokens if t.type == "file_content" and not t.done
        )
        assert path == "/a.py"
        assert search == "old"
        assert content == "new"


class TestInterleaved:
    def test_multiple_calls_get_distinct_emission_indices(self):
        """Each ToolCallStart bumps the per-turn emission_index counter."""
        py_args = json.dumps({"title": "t", "thinking": "T", "code": "x"})
        file_args = json.dumps({"path": "/a.py", "content": "hi"})

        events = [
            ToolCallStart("c1", TOOL_WRITE_FILE),
            ToolCallArgDelta("c1", file_args),
            ToolCallEnd("c1"),
            ToolCallStart("c2", TOOL_PYTHON),
            ToolCallArgDelta("c2", py_args),
            ToolCallEnd("c2"),
        ]
        tokens = _tokens(events)

        # File emission is index 0; python tokens are index 1.
        emission_token = next(t for t in tokens if t.type == "emission")
        python_tokens = [t for t in tokens if t.type == "python"]
        assert emission_token.emission_index == 0
        assert all(t.emission_index == 1 for t in python_tokens)

    def test_unknown_call_id_ignored(self):
        tokens = _tokens(
            [
                ToolCallArgDelta("ghost", '{"x": 1}'),
                ToolCallEnd("ghost"),
            ]
        )
        assert tokens == []


class TestTextPart:
    def test_textpart_becomes_text_emission(self):
        from agex.agent.emissions import TextEmission
        from agex.llm.formats.tool_use.events import TextPart

        tokens = _tokens([TextPart(text="plain reply")])
        emission_tokens = [t for t in tokens if t.type == "emission"]
        assert len(emission_tokens) == 1
        emission = emission_tokens[0].emission
        assert isinstance(emission, TextEmission)
        assert emission.text == "plain reply"

    def test_empty_textpart_dropped(self):
        from agex.llm.formats.tool_use.events import TextPart

        assert _tokens([TextPart(text="")]) == []


@pytest.mark.asyncio
async def test_async_parser():
    from agex.llm.formats.tool_use import aparse_tool_events

    args = json.dumps({"title": "t", "thinking": "T", "code": "x"})

    async def stream():
        yield ToolCallStart("c1", TOOL_PYTHON)
        for chunk in _chunked("c1", args, 5):
            yield chunk
        yield ToolCallEnd("c1")

    out: list[TokenChunk] = []
    async for t in aparse_tool_events(stream()):
        out.append(t)

    code_content = "".join(t.content for t in out if t.type == "python" and not t.done)
    assert code_content == "x"
