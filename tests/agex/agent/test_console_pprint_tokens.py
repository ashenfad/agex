"""Tests for pprint_tokens — rendering of streamed TokenChunks,
including the tool-use ``file_action`` type and XML-mode edit-body
suppression."""

import io

from agex.agent.console import pprint_tokens
from agex.agent.datatypes import EditAction, FileAction
from agex.llm.core import StreamToken


def _print(token):
    """Run pprint_tokens against an in-memory stream and return output."""
    buf = io.StringIO()
    pprint_tokens(token, stream=buf, color="never")
    return buf.getvalue()


class TestFileActionRendering:
    """Tool-use wire formats emit a single file_action token carrying a
    built FileAction/EditAction. The helper should render a readable
    summary."""

    def test_write_file_create(self):
        token = StreamToken(
            type="file_action",
            content="",
            done=True,
            action=FileAction(path="/helpers/x.py", content="X = 1"),
        )
        out = _print(token)
        assert "📁" in out
        assert "/helpers/x.py" in out
        assert "[CREATE]" in out

    def test_write_file_append(self):
        token = StreamToken(
            type="file_action",
            content="",
            done=True,
            action=FileAction(path="/x.py", content="more", mode="append"),
        )
        out = _print(token)
        assert "[APPEND]" in out

    def test_edit_replace(self):
        token = StreamToken(
            type="file_action",
            content="",
            done=True,
            action=EditAction(
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
            type="file_action",
            content="",
            done=True,
            action=EditAction(
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

    def test_no_raw_xml_tags_emitted(self):
        """Under no circumstances should the tool-use file_action path
        print XML-shaped tags like <SEARCH> or <INSERT-AFTER>."""
        token = StreamToken(
            type="file_action",
            content="",
            done=True,
            action=EditAction(
                path="/a.py",
                search="x",
                content="y",
                operation="insert-before",
            ),
        )
        out = _print(token)
        for tag in ("<SEARCH>", "<REPLACE>", "<INSERT-AFTER>", "<INSERT-BEFORE>"):
            assert tag not in out


class TestXmlEditBodySuppressed:
    """The XML wire format emits the raw inline body (<SEARCH>...</SEARCH>
    <REPLACE>...</REPLACE>) as the second ``edit`` token. The helper
    should print the header summary but suppress the raw tags."""

    def test_edit_header_rendered(self):
        token = StreamToken(
            type="edit",
            content="path=/a.py,match_all=False",
            done=False,
        )
        out = _print(token)
        assert "✏️" in out
        assert "/a.py" in out
        assert "[EDIT]" in out

    def test_edit_body_xml_tags_suppressed(self):
        token = StreamToken(
            type="edit",
            content="<SEARCH>old</SEARCH><REPLACE>new</REPLACE>",
            done=False,
        )
        out = _print(token)
        assert "<SEARCH>" not in out
        assert "<REPLACE>" not in out
        assert out == ""

    def test_edit_body_insert_tags_suppressed(self):
        token = StreamToken(
            type="edit",
            content="<SEARCH>x</SEARCH><INSERT-AFTER>y</INSERT-AFTER>",
            done=False,
        )
        out = _print(token)
        assert "<INSERT-AFTER>" not in out


class TestUnaffectedPaths:
    def test_thinking_content_prints(self):
        token = StreamToken(type="thinking", content="reasoning...", done=False)
        out = _print(token)
        assert "reasoning..." in out

    def test_file_header_still_rendered(self):
        """XML-mode file token header is unchanged."""
        token = StreamToken(
            type="file",
            content="path=foo.py,mode=write",
            done=False,
        )
        out = _print(token)
        assert "📁" in out
        assert "foo.py" in out
        assert "[CREATE]" in out

    def test_done_emits_trailing_newline(self):
        token = StreamToken(type="thinking", content="", done=True)
        out = _print(token)
        assert out == "\n"
