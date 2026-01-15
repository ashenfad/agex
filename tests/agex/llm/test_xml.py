"""Tests for XML parsing utilities (LLM-specific)."""

import pytest

from agex.llm.core import ResponseParseError
from agex.llm.xml import (
    TokenChunk,
    XMLResponse,
    parse_xml_response,
    tokenize_xml_stream,
)


class TestParseXMLResponse:
    """Tests for parse_xml_response()."""

    def test_basic_parsing(self):
        """Test parsing basic XML response."""
        xml = """
        <THINKING>
        I need to calculate the sum.
        </THINKING>
        <PYTHON>
        total = sum(numbers)
        task_success(total)
        </PYTHON>
        """
        result = parse_xml_response(xml)
        assert result.thinking == "I need to calculate the sum."
        # The code content preserves whitespace as written
        assert "total = sum(numbers)" in result.code
        assert "task_success(total)" in result.code
        assert result.title == ""

    def test_case_insensitive_tags(self):
        """Test that tags are case-insensitive."""
        xml = """
        <thinking>
        My reasoning here.
        </thinking>
        <python>
        my_code()
        </python>
        """
        result = parse_xml_response(xml)
        assert result.thinking == "My reasoning here."
        assert result.code == "my_code()"

    def test_mixed_case_tags(self):
        """Test mixed case tags."""
        xml = """
        <Thinking>
        Mixed case reasoning.
        </Thinking>
        <Python>
        mixed_case_code()
        </Python>
        """
        result = parse_xml_response(xml)
        assert result.thinking == "Mixed case reasoning."
        assert result.code == "mixed_case_code()"

    def test_with_title(self):
        """Test parsing with optional title tag."""
        xml = """
        <TITLE>Calculating sum</TITLE>
        <THINKING>
        I need to add numbers.
        </THINKING>
        <PYTHON>
        result = a + b
        </PYTHON>
        """
        result = parse_xml_response(xml)
        assert result.title == "Calculating sum"
        assert result.thinking == "I need to add numbers."
        assert result.code == "result = a + b"

    def test_multiline_content(self):
        """Test parsing with multiline content."""
        xml = """
        <THINKING>
        First, I'll check the input.
        Then, I'll process it.
        Finally, I'll return the result.
        </THINKING>
        <PYTHON>
        if not data:
            return None
        
        processed = process(data)
        return processed
        </PYTHON>
        """
        result = parse_xml_response(xml)
        assert "First, I'll check" in result.thinking
        assert "Then, I'll process" in result.thinking
        assert "Finally, I'll return" in result.thinking
        assert "if not data:" in result.code
        assert "return processed" in result.code

    def test_missing_thinking_tag(self):
        """Test error when THINKING tag is missing."""
        xml = "<PYTHON>some_code()</PYTHON>"
        with pytest.raises(ResponseParseError) as exc_info:
            parse_xml_response(xml)
        assert "Missing <THINKING> tags" in str(exc_info.value)

    def test_missing_code_tag(self):
        """Test error when CODE tag is missing."""
        xml = "<THINKING>some thinking</THINKING>"
        with pytest.raises(ResponseParseError) as exc_info:
            parse_xml_response(xml)
        assert "Missing <PYTHON> tags" in str(exc_info.value)

    def test_empty_tags(self):
        """Test parsing with empty tags."""
        xml = """
        <THINKING></THINKING>
        <PYTHON></PYTHON>
        """
        result = parse_xml_response(xml)
        assert result.thinking == ""
        assert result.code == ""

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is stripped."""
        xml = """
        <THINKING>
        
            My thinking with whitespace.
            
        </THINKING>
        <PYTHON>
        
            my_code()
            
        </PYTHON>
        """
        result = parse_xml_response(xml)
        # Content should be stripped
        assert result.thinking.startswith("My thinking")
        assert result.code.startswith("my_code")

    def test_content_with_similar_tags(self):
        """Test that content containing tag-like strings doesn't break parsing."""
        xml = """
        <THINKING>
        I need to use <PYTHON> tags in my explanation.
        </THINKING>
        <PYTHON>
        print("<THINKING> is not real code")
        </PYTHON>
        """
        result = parse_xml_response(xml)
        # Should only match the outer tags
        assert "<PYTHON> tags" in result.thinking
        assert "<THINKING>" in result.code

    def test_parsing_with_mode(self):
        """Test parsing with mode attribute in FILE tags."""
        xml = """
        <THINKING>reasoning</THINKING>
        <FILE path="test.txt" mode="append">Line 2</FILE>
        <FILE path="new.txt">New File</FILE>
        <PYTHON>pass</PYTHON>
        """
        result = parse_xml_response(xml)
        assert result.files["test.txt"] == "Line 2"
        assert result.file_modes["test.txt"] == "append"
        assert result.files["new.txt"] == "New File"
        assert result.file_modes["new.txt"] == "write"


class TestTokenizeXMLStream:
    """Tests for tokenize_xml_stream()."""

    def test_simple_stream(self):
        """Test tokenizing a simple complete stream."""
        chunks = [
            "<TITLE>",
            "Calculating sum",
            "</TITLE>",
            "<THINKING>",
            "I will ",
            "calculate ",
            "the sum",
            "</THINKING>",
            "<PYTHON>",
            "result = sum(nums)",
            "</PYTHON>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        title_tokens = [t for t in tokens if t.type == "title"]
        thinking_tokens = [t for t in tokens if t.type == "thinking"]
        code_tokens = [t for t in tokens if t.type == "python"]

        assert len(title_tokens) > 0
        assert len(thinking_tokens) > 0
        assert len(code_tokens) > 0

        assert any(t.done for t in title_tokens)
        assert any(t.done for t in thinking_tokens)
        assert any(t.done for t in code_tokens)

        title_content = "".join(t.content for t in title_tokens if not t.done)
        thinking_content = "".join(t.content for t in thinking_tokens if not t.done)
        code_content = "".join(t.content for t in code_tokens if not t.done)

        assert "Calculating" in title_content
        assert "calculate" in thinking_content
        assert "result = sum(nums)" in code_content

    def test_split_tags(self):
        """Test handling tags split across chunks."""
        chunks = [
            "<THI",
            "NKING>",
            "Content here",
            "</THIN",
            "KING>",
            "<PY",
            "THON>",
            "code_here()",
            "</PYTHON>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        thinking_tokens = [t for t in tokens if t.type == "thinking" and not t.done]
        code_tokens = [t for t in tokens if t.type == "python" and not t.done]

        thinking_content = "".join(t.content for t in thinking_tokens)
        code_content = "".join(t.content for t in code_tokens)

        assert "Content here" in thinking_content
        assert "code_here()" in code_content

    def test_case_insensitive_tags(self):
        """Test that streaming handles case-insensitive tags."""
        chunks = [
            "<thinking>",
            "lowercase tags",
            "</thinking>",
            "<python>",
            "python_code()",
            "</python>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        thinking_tokens = [t for t in tokens if t.type == "thinking" and not t.done]
        code_tokens = [t for t in tokens if t.type == "python" and not t.done]

        thinking_content = "".join(t.content for t in thinking_tokens)
        code_content = "".join(t.content for t in code_tokens)

        assert "lowercase tags" in thinking_content
        assert "python_code()" in code_content

    def test_incremental_content(self):
        """Test that content is yielded incrementally."""
        chunks = [
            "<THINKING>",
            "First part. ",
            "Second part. ",
            "Third part.",
            "</THINKING>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        thinking_tokens = [t for t in tokens if t.type == "thinking"]

        # Should have multiple content chunks before the done marker
        content_chunks = [t for t in thinking_tokens if not t.done]
        assert len(content_chunks) > 0

        # Should end with done marker
        assert thinking_tokens[-1].done
        assert thinking_tokens[-1].content == ""

    def test_empty_sections(self):
        """Test handling empty sections."""
        chunks = ["<THINKING>", "</THINKING>", "<PYTHON>", "</PYTHON>"]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        # Should still get done markers even for empty sections
        thinking_done = [t for t in tokens if t.type == "thinking" and t.done]
        code_done = [t for t in tokens if t.type == "python" and t.done]

        assert len(thinking_done) == 1
        assert len(code_done) == 1

    def test_multiline_content(self):
        """Test handling multiline content."""
        chunks = [
            "<THINKING>",
            "Line 1\n",
            "Line 2\n",
            "Line 3",
            "</THINKING>",
            "<PYTHON>",
            "def func():\n",
            "    pass",
            "</PYTHON>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        thinking_content = "".join(
            t.content for t in tokens if t.type == "thinking" and not t.done
        )
        code_content = "".join(
            t.content for t in tokens if t.type == "python" and not t.done
        )

        assert "Line 1\nLine 2\nLine 3" in thinking_content
        assert "def func():" in code_content
        assert "    pass" in code_content

    def test_large_chunks(self):
        """Test handling large chunks with complete sections."""
        chunks = [
            "<THINKING>This is a long thinking section with lots of text.</THINKING>"
            "<PYTHON>long_code()\nmore_code()\neven_more()</PYTHON>"
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        thinking_tokens = [t for t in tokens if t.type == "thinking"]
        code_tokens = [t for t in tokens if t.type == "python"]

        # Should successfully parse even when everything comes in one chunk
        assert len(thinking_tokens) > 0
        assert len(code_tokens) > 0
        assert any(t.done for t in thinking_tokens)
        assert any(t.done for t in code_tokens)

    def test_content_order_preserved(self):
        """Test that content order is preserved through tokenization."""
        chunks = [
            "<THINKING>",
            "Part A. ",
            "Part B. ",
            "Part C.",
            "</THINKING>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        content_tokens = [t for t in tokens if t.type == "thinking" and not t.done]
        full_content = "".join(t.content for t in content_tokens)

        # Order should be preserved
        assert full_content.index("Part A") < full_content.index("Part B")
        assert full_content.index("Part B") < full_content.index("Part C")

    def test_closing_tag_not_in_content(self):
        """Test that closing tags split across chunks don't appear in content."""
        # This tests the bug where "</THINKING>" was appearing in output
        chunks = [
            "<THINKING>",
            "Some content here<",  # Buffer ends with potential tag start
            "/THINKING>",  # Closing tag continues
            "<PYTHON>code()</PYTHON>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        # Collect all content
        all_content = "".join(t.content for t in tokens if not t.done)

        # Tags should NOT appear in content
        assert "</THINKING>" not in all_content
        assert "<PYTHON>" not in all_content
        assert "</PYTHON>" not in all_content

        # But actual content should be there
        assert "Some content here" in all_content
        assert "code()" in all_content

    def test_streaming_with_mode(self):
        """Test streaming with mode attribute in FILE tags."""
        chunks = [
            "<THINKING>reasoning</THINKING>",
            '<FILE path="stream.py" mode="append">',
            "# appended code",
            "</FILE>",
            "<PYTHON>pass</PYTHON>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        file_path_token = next(
            t for t in tokens if t.type == "file" and t.content.startswith("path=")
        )
        assert file_path_token.content == "path=stream.py,mode=append"


class TestXMLResponse:
    """Tests for XMLResponse dataclass."""

    def test_basic_creation(self):
        """Test creating XMLResponse."""
        response = XMLResponse(thinking="My thinking", code="my_code()")
        assert response.thinking == "My thinking"
        assert response.code == "my_code()"
        assert response.title == ""

    def test_with_title(self):
        """Test creating XMLResponse with title."""
        response = XMLResponse(
            thinking="My thinking", code="my_code()", title="My title"
        )
        assert response.title == "My title"


class TestTokenChunk:
    """Tests for TokenChunk dataclass."""

    def test_basic_creation(self):
        """Test creating TokenChunk."""
        chunk = TokenChunk(type="thinking", content="Hello")
        assert chunk.type == "thinking"
        assert chunk.content == "Hello"
        assert chunk.done is False

    def test_title_chunk(self):
        """Test creating a title TokenChunk."""
        chunk = TokenChunk(type="title", content="Summarizing task")
        assert chunk.type == "title"
        assert chunk.content == "Summarizing task"
        assert chunk.done is False

    def test_done_marker(self):
        """Test creating done marker TokenChunk."""
        chunk = TokenChunk(type="python", content="", done=True)
        assert chunk.type == "python"
        assert chunk.content == ""
        assert chunk.done is True

    def test_stop_after_first_python_section(self):
        """Test that tokenization stops after the first </PYTHON> tag."""
        chunks = [
            "<TITLE>First Title</TITLE>",
            "<THINKING>First Reasoning</THINKING>",
            "<PYTHON>first_code()</PYTHON>",
            "<TITLE>Second Title</TITLE>",
            "<THINKING>Second Reasoning</THINKING>",
            "<PYTHON>second_code()</PYTHON>",
        ]
        tokens = list(tokenize_xml_stream(iter(chunks)))

        # Should only contain tokens for the first section
        titles = [t.content for t in tokens if t.type == "title" and not t.done]
        thinking = [t.content for t in tokens if t.type == "thinking" and not t.done]
        python = [t.content for t in tokens if t.type == "python" and not t.done]

        assert titles == ["First Title"]
        assert thinking == ["First Reasoning"]
        assert python == ["first_code()"]

        # Should NOT contain content from the second section
        assert "Second Title" not in str(tokens)
        assert "Second Reasoning" not in str(tokens)
        assert "second_code()" not in str(tokens)
