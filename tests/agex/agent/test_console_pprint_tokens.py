"""Tests for pprint_tokens — rendering of streamed TokenChunks."""

import io

from agex.agent.console import pprint_tokens
from agex.agent.emissions import FileEditEmission, FileWriteEmission
from agex.llm.core import StreamToken


def _print(token):
    """Run pprint_tokens against an in-memory stream and return output."""
    buf = io.StringIO()
    pprint_tokens(token, stream=buf, color="never")
    return buf.getvalue()


def _print_all(tokens):
    buf = io.StringIO()
    for token in tokens:
        pprint_tokens(token, stream=buf, color="never")
    return buf.getvalue()


class TestFileStreamRendering:
    """File tool args stream as ``file_path`` / ``file_search`` /
    ``file_content`` tokens before the final ``emission`` token.  The
    helper should render the path, content, and a mode/operation
    trailer so debuggers can see what the agent wrote."""

    def test_write_file_create(self):
        emission = FileWriteEmission(path="/helpers/x.py", content="X = 1")
        tokens = [
            StreamToken(type="file_path", content="/helpers/x.py", start=True),
            StreamToken(type="file_path", content="", done=True),
            StreamToken(type="file_content", content="X = 1", start=True),
            StreamToken(type="file_content", content="", done=True),
            StreamToken(type="emission", done=True, emission=emission),
        ]
        out = _print_all(tokens)
        assert "📁" in out
        assert "/helpers/x.py" in out
        assert "X = 1" in out
        assert "create" in out

    def test_write_file_append(self):
        emission = FileWriteEmission(path="/x.py", content="more", mode="append")
        tokens = [
            StreamToken(type="file_path", content="/x.py", start=True),
            StreamToken(type="file_path", content="", done=True),
            StreamToken(type="file_content", content="more", start=True),
            StreamToken(type="file_content", content="", done=True),
            StreamToken(type="emission", done=True, emission=emission),
        ]
        out = _print_all(tokens)
        assert "append" in out

    def test_edit_replace(self):
        emission = FileEditEmission(
            path="/a.py",
            search="old",
            content="new",
            operation="replace",
        )
        tokens = [
            StreamToken(type="file_path", content="/a.py", start=True),
            StreamToken(type="file_path", content="", done=True),
            StreamToken(type="file_search", content="old", start=True),
            StreamToken(type="file_search", content="", done=True),
            StreamToken(type="file_content", content="new", start=True),
            StreamToken(type="file_content", content="", done=True),
            StreamToken(type="emission", done=True, emission=emission),
        ]
        out = _print_all(tokens)
        assert "/a.py" in out
        assert "old" in out
        assert "new" in out
        assert "replace" in out

    def test_edit_insert_after_with_match_all(self):
        emission = FileEditEmission(
            path="/a.py",
            search="x",
            content="y",
            operation="insert-after",
            match_all=True,
        )
        tokens = [
            StreamToken(type="file_path", content="/a.py", start=True),
            StreamToken(type="file_path", content="", done=True),
            StreamToken(type="file_search", content="x", start=True),
            StreamToken(type="file_search", content="", done=True),
            StreamToken(type="file_content", content="y", start=True),
            StreamToken(type="file_content", content="", done=True),
            StreamToken(type="emission", done=True, emission=emission),
        ]
        out = _print_all(tokens)
        assert "insert-after" in out
        assert "match_all" in out


class TestUnaffectedPaths:
    def test_thinking_content_prints(self):
        token = StreamToken(type="thinking", content="reasoning...", done=False)
        out = _print(token)
        assert "reasoning..." in out

    def test_done_emits_trailing_newline(self):
        token = StreamToken(type="thinking", content="", done=True)
        out = _print(token)
        assert out == "\n"

    def test_text_block_printed_with_prefix_on_first_chunk(self):
        """User-facing prose (the new ``text`` token) gets a distinctive
        emoji prefix at section start so it reads as a message rather
        than code or thinking."""
        token = StreamToken(
            type="text",
            content="Working on it...",
            done=False,
            start=True,
        )
        out = _print(token)
        assert "💬" in out
        assert "Working on it..." in out
