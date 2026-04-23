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


class TestFileEmissionRendering:
    """Tool-use wire formats emit a single ``emission`` token carrying a
    fully built :class:`FileWriteEmission` / :class:`FileEditEmission`.
    The helper should render a readable summary."""

    def test_write_file_create(self):
        token = StreamToken(
            type="emission",
            content="",
            done=True,
            emission=FileWriteEmission(path="/helpers/x.py", content="X = 1"),
        )
        out = _print(token)
        assert "📁" in out
        assert "/helpers/x.py" in out
        assert "[CREATE]" in out

    def test_write_file_append(self):
        token = StreamToken(
            type="emission",
            content="",
            done=True,
            emission=FileWriteEmission(path="/x.py", content="more", mode="append"),
        )
        out = _print(token)
        assert "[APPEND]" in out

    def test_edit_replace(self):
        token = StreamToken(
            type="emission",
            content="",
            done=True,
            emission=FileEditEmission(
                path="/a.py",
                search="old",
                content="new",
                operation="replace",
            ),
        )
        out = _print(token)
        assert "✏️" in out
        assert "/a.py" in out
        assert "[EDIT]" in out
        assert "replace" in out

    def test_edit_insert_after_with_match_all(self):
        token = StreamToken(
            type="emission",
            content="",
            done=True,
            emission=FileEditEmission(
                path="/a.py",
                search="x",
                content="y",
                operation="insert-after",
                match_all=True,
            ),
        )
        out = _print(token)
        assert "[EDIT ALL]" in out
        assert "insert-after" in out


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
