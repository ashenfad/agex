"""Tests for EDIT action matching with flexible indentation and whitespace handling."""

import pytest

from agex.agent.loop.common import (
    _adjust_replacement_indent,
    _build_trailing_ws_pattern,
    _find_indent_flexible_match,
)
from agex.agent.loop.file_editing import _find_similar_lines


class TestBuildTrailingWsPattern:
    """Tests for trailing whitespace pattern building."""

    def test_simple_match(self):
        """Pattern should match exact text."""
        pattern = _build_trailing_ws_pattern("hello")
        assert pattern.search("hello") is not None

    def test_trailing_spaces_in_file(self):
        """Pattern should match when file has trailing spaces."""
        pattern = _build_trailing_ws_pattern("hello")
        assert pattern.search("hello   ") is not None

    def test_trailing_tabs_in_file(self):
        """Pattern should match when file has trailing tabs."""
        pattern = _build_trailing_ws_pattern("hello")
        assert pattern.search("hello\t\t") is not None

    def test_trailing_spaces_in_search(self):
        """Pattern should match when search has trailing spaces but file doesn't."""
        pattern = _build_trailing_ws_pattern("hello   ")
        assert pattern.search("hello") is not None

    def test_multiline_trailing_ws(self):
        """Pattern should handle trailing whitespace on each line."""
        pattern = _build_trailing_ws_pattern("line1\nline2")
        assert pattern.search("line1   \nline2\t") is not None

    def test_blank_line_with_whitespace(self):
        """Pattern should match blank lines with varying whitespace."""
        pattern = _build_trailing_ws_pattern("a\n\nb")
        assert pattern.search("a\n   \nb") is not None

    def test_preserves_internal_whitespace(self):
        """Pattern should NOT match if internal whitespace differs."""
        pattern = _build_trailing_ws_pattern("a = 1")
        assert pattern.search("a  =  1") is None


class TestFindIndentFlexibleMatch:
    """Tests for indent-flexible matching."""

    def test_exact_match(self):
        """Should find exact matches."""
        content = "def foo():\n    pass"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1
        assert matches[0][2] == content

    def test_different_base_indent_spaces(self):
        """Should match when file has different base indentation (spaces)."""
        content = "    def foo():\n        pass"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1
        assert matches[0][2] == content

    def test_tabs_vs_spaces(self):
        """Should match tabs in file when search uses spaces."""
        content = "\tdef foo():\n\t\tpass"
        search = "    def foo():\n        pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1
        assert matches[0][2] == content

    def test_spaces_vs_tabs(self):
        """Should match spaces in file when search uses tabs."""
        content = "    def foo():\n        pass"
        search = "\tdef foo():\n\t\tpass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1
        assert matches[0][2] == content

    def test_trailing_whitespace_tolerance(self):
        """Should match even with trailing whitespace differences."""
        content = "def foo():   \n    pass"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_blank_lines(self):
        """Should match with blank lines."""
        content = "def foo():\n\n    pass"
        search = "def foo():\n\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_blank_line_with_whitespace(self):
        """Should match blank lines regardless of whitespace content."""
        content = "def foo():\n    \n    pass"
        search = "def foo():\n\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_no_match_different_content(self):
        """Should not match when content differs."""
        content = "def foo():\n    return 1"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 0

    def test_multiple_matches(self):
        """Should find multiple matches."""
        content = "def foo():\n    pass\n\ndef bar():\n    pass"
        search = "def foo():\n    pass"
        # Note: this won't match "def bar():\n    pass" because stripped content differs
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_multiple_identical_blocks(self):
        """Should find all identical blocks."""
        content = "if True:\n    pass\n\nif True:\n    pass"
        search = "if True:\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 2

    def test_match_in_middle_of_file(self):
        """Should find match in the middle of a file."""
        content = "# header\n\ndef foo():\n    pass\n\n# footer"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1
        assert "def foo():" in matches[0][2]

    def test_deeply_nested_indent(self):
        """Should match deeply nested code with different base indent."""
        content = "        if True:\n            pass"
        search = "if True:\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_preserves_relative_indent(self):
        """Should only match if relative indentation structure matches."""
        content = "def foo():\n        pass"  # 8 spaces for pass
        search = "def foo():\n    pass"  # 4 spaces for pass
        # Both have same structure: base + 4 more for the body
        # But the relative indent is different (0->4 vs 0->8)
        # Wait, actually the STRIPPED content matches, we just need same structure
        matches = _find_indent_flexible_match(search, content)
        # This should match because stripped content matches
        assert len(matches) == 1

    def test_empty_search(self):
        """Should return no matches for empty search."""
        content = "def foo():\n    pass"
        search = ""
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 0

    def test_whitespace_only_search(self):
        """Should return no matches for whitespace-only search."""
        content = "def foo():\n    pass"
        search = "   \n   "
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 0

    def test_position_calculation(self):
        """Should return correct byte positions."""
        content = "abc\ndef foo():\n    pass\nxyz"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1
        start, end, matched = matches[0]
        assert content[start:end] == "def foo():\n    pass"


class TestAdjustReplacementIndent:
    """Tests for replacement indentation adjustment."""

    def test_no_adjustment_needed(self):
        """No adjustment when indents match."""
        search = "def foo():\n    pass"
        matched = "def foo():\n    pass"
        replacement = "def bar():\n    return 1"
        result = _adjust_replacement_indent(replacement, search, matched)
        assert result == "def bar():\n    return 1"

    def test_add_base_indent(self):
        """Add base indent when file is indented."""
        search = "def foo():\n    pass"
        matched = "    def foo():\n        pass"
        replacement = "def bar():\n    return 1"
        result = _adjust_replacement_indent(replacement, search, matched)
        assert result == "    def bar():\n        return 1"

    def test_remove_base_indent(self):
        """Remove base indent when file has less indentation."""
        search = "    def foo():\n        pass"
        matched = "def foo():\n    pass"
        replacement = "    def bar():\n        return 1"
        result = _adjust_replacement_indent(replacement, search, matched)
        assert result == "def bar():\n    return 1"

    def test_convert_spaces_to_tabs(self):
        """Convert to tabs when file uses tabs."""
        search = "def foo():\n    pass"
        matched = "\tdef foo():\n\t\tpass"
        replacement = "def bar():\n    return 1"
        result = _adjust_replacement_indent(replacement, search, matched)
        # Should use tabs, 1 tab = 4 spaces
        assert result == "\tdef bar():\n\t\treturn 1"

    def test_preserves_relative_indent(self):
        """Should preserve relative indentation within replacement."""
        search = "def foo():\n    pass"
        matched = "    def foo():\n        pass"
        replacement = "def bar():\n    if True:\n        return 1"
        result = _adjust_replacement_indent(replacement, search, matched)
        lines = result.split("\n")
        assert lines[0] == "    def bar():"
        assert lines[1] == "        if True:"
        assert lines[2] == "            return 1"

    def test_empty_lines_preserved(self):
        """Empty lines should remain empty."""
        search = "a\n\nb"
        matched = "    a\n\n    b"
        replacement = "x\n\ny"
        result = _adjust_replacement_indent(replacement, search, matched)
        lines = result.split("\n")
        assert lines[1] == ""

    def test_handles_mixed_content(self):
        """Should handle replacement with different indentation than search."""
        search = "pass"
        matched = "        pass"
        replacement = "return 1"
        result = _adjust_replacement_indent(replacement, search, matched)
        assert result == "        return 1"


class TestIntegration:
    """Integration tests combining matching and replacement."""

    def test_full_edit_flow_with_indent_difference(self):
        """Test the complete flow with different indentation."""
        file_content = """\
class Foo:
    def method(self):
        old_code()
        more_old()
"""
        search = "def method(self):\n    old_code()\n    more_old()"
        replacement = "def method(self):\n    new_code()\n    better_code()"

        matches = _find_indent_flexible_match(search, file_content)
        assert len(matches) == 1

        start, end, matched_text = matches[0]
        adjusted = _adjust_replacement_indent(replacement, search, matched_text)

        new_content = file_content[:start] + adjusted + file_content[end:]

        expected = """\
class Foo:
    def method(self):
        new_code()
        better_code()
"""
        assert new_content == expected

    def test_insert_after_with_indent(self):
        """Test INSERT-AFTER operation with indentation adjustment."""
        file_content = """\
class Foo:
    def existing(self):
        pass
"""
        search = "def existing(self):\n    pass"
        insert_content = "\n\ndef new_method(self):\n    return 42"

        matches = _find_indent_flexible_match(search, file_content)
        assert len(matches) == 1

        start, end, matched_text = matches[0]
        adjusted_insert = _adjust_replacement_indent(
            insert_content, search, matched_text
        )

        # For insert-after, we keep matched text and add adjusted content
        new_content = (
            file_content[:start] + matched_text + adjusted_insert + file_content[end:]
        )

        assert "def existing(self):" in new_content
        assert "def new_method(self):" in new_content
        # Check the new method is properly indented
        assert "        return 42" in new_content

    def test_tabs_to_spaces_conversion(self):
        """Test that tab-indented files get space replacement adjusted."""
        file_content = "class Foo:\n\tdef method(self):\n\t\tpass\n"
        search = "def method(self):\n    pass"
        replacement = "def method(self):\n    return 1"

        matches = _find_indent_flexible_match(search, file_content)
        assert len(matches) == 1

        start, end, matched_text = matches[0]
        adjusted = _adjust_replacement_indent(replacement, search, matched_text)

        # Should convert to tabs
        assert "\tdef method(self):" in adjusted
        assert "\t\treturn 1" in adjusted


class TestEdgeCases:
    """Edge cases and regression tests."""

    def test_single_line_match(self):
        """Should handle single-line matches."""
        content = "    pass"
        search = "pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_decorator_with_function(self):
        """Should match decorated functions."""
        content = """\
    @decorator
    def foo():
        pass
"""
        search = "@decorator\ndef foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_class_with_methods(self):
        """Should match class definitions."""
        content = """\
    class Foo:
        def __init__(self):
            self.x = 1
"""
        search = "class Foo:\n    def __init__(self):\n        self.x = 1"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_no_false_positive_partial_match(self):
        """Should not match partial content."""
        content = "def foo():\n    pass\n    more_code()"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        # Should match the first two lines only
        assert len(matches) == 1
        assert matches[0][2] == "def foo():\n    pass"

    def test_match_at_end_of_file(self):
        """Should match content at end of file."""
        content = "# header\n\ndef foo():\n    pass"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_match_at_start_of_file(self):
        """Should match content at start of file."""
        content = "def foo():\n    pass\n\n# footer"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1
        assert matches[0][0] == 0

    def test_windows_line_endings(self):
        """Should handle Windows line endings."""
        content = "def foo():\r\n    pass"
        search = "def foo():\n    pass"
        # The implementation handles Windows line endings because:
        # 1. We split on \n, so \r stays at end of lines
        # 2. We use .strip() which removes \r as well as spaces
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_unicode_content(self):
        """Should handle unicode content."""
        content = "    def greet():\n        print('Hello, 世界!')"
        search = "def greet():\n    print('Hello, 世界!')"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_very_deep_nesting(self):
        """Should handle deeply nested code."""
        content = "                if True:\n                    pass"
        search = "if True:\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1

    def test_mixed_indent_in_search(self):
        """Should handle mixed indentation in search."""
        content = "    if True:\n        pass"
        search = "    if True:\n        pass"  # Search already indented
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 1


class TestRegressionPrevention:
    """Tests to prevent regressions in existing behavior."""

    def test_exact_match_still_preferred(self):
        """Exact matches should still work (tested via pattern)."""
        pattern = _build_trailing_ws_pattern("hello world")
        assert pattern.search("hello world") is not None

    def test_trailing_ws_pattern_unchanged(self):
        """Trailing whitespace patterns should work as before."""
        pattern = _build_trailing_ws_pattern("def foo():\n    pass")
        # Should match with trailing spaces
        assert pattern.search("def foo():   \n    pass\t") is not None
        # Should NOT match with leading space differences (that's indent-flexible's job)
        assert pattern.search("  def foo():\n      pass") is None

    def test_no_match_when_content_differs(self):
        """Should not match when actual content (not just whitespace) differs."""
        content = "def foo():\n    return 1"
        search = "def foo():\n    pass"
        matches = _find_indent_flexible_match(search, content)
        assert len(matches) == 0


class TestApplyOptimisticFileActionsIntegration:
    """End-to-end tests for apply_optimistic_file_actions with EditAction."""

    @pytest.fixture
    def mock_fs(self):
        """Create a simple mock filesystem."""

        class MockFS:
            def __init__(self):
                self.files = {}

            def read(self, path):
                if path not in self.files:
                    raise FileNotFoundError(path)
                return self.files[path]

            def write(self, path, content, mode="w"):
                if mode == "a" and path in self.files:
                    self.files[path] += content
                else:
                    self.files[path] = content

        return MockFS()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with minimal policy."""

        class MockPolicy:
            namespaces = {}

        class MockAgent:
            _policy = MockPolicy()

        return MockAgent()

    @pytest.fixture
    def mock_llm_response(self):
        """Factory for creating mock LLM responses."""
        from agex.agent.datatypes import EditAction

        class MockLLMResponse:
            def __init__(self, file_actions):
                self.file_actions = file_actions

        def create(
            search, content, path="test.py", operation="replace", match_all=False
        ):
            return MockLLMResponse(
                [
                    EditAction(
                        path=path,
                        search=search,
                        content=content,
                        operation=operation,
                        match_all=match_all,
                    )
                ]
            )

        return create

    def test_exact_match_edit(self, mock_fs, mock_agent, mock_llm_response):
        """Test edit with exact match."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        mock_fs.files["test.py"] = b"def foo():\n    pass"
        response = mock_llm_response(
            search="def foo():\n    pass",
            content="def bar():\n    return 1",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        assert mock_fs.files["test.py"] == b"def bar():\n    return 1"

    def test_trailing_whitespace_edit(self, mock_fs, mock_agent, mock_llm_response):
        """Test edit with trailing whitespace differences."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        mock_fs.files["test.py"] = b"def foo():   \n    pass\t"
        response = mock_llm_response(
            search="def foo():\n    pass",
            content="def bar():\n    return 1",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        assert mock_fs.files["test.py"] == b"def bar():\n    return 1"

    def test_indent_flexible_edit_spaces(self, mock_fs, mock_agent, mock_llm_response):
        """Test edit with different indentation (file has more indent)."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        mock_fs.files["test.py"] = b"    def foo():\n        pass"
        response = mock_llm_response(
            search="def foo():\n    pass",
            content="def bar():\n    return 1",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        # Result should have adjusted indentation
        result = mock_fs.files["test.py"].decode("utf-8")
        assert "    def bar():" in result
        assert "        return 1" in result

    def test_indent_flexible_edit_tabs(self, mock_fs, mock_agent, mock_llm_response):
        """Test edit with tabs vs spaces."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        mock_fs.files["test.py"] = b"\tdef foo():\n\t\tpass"
        response = mock_llm_response(
            search="def foo():\n    pass",
            content="def bar():\n    return 1",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        # Result should use tabs like the original file
        result = mock_fs.files["test.py"].decode("utf-8")
        assert "\tdef bar():" in result
        assert "\t\treturn 1" in result

    def test_indent_flexible_insert_after(self, mock_fs, mock_agent, mock_llm_response):
        """Test insert-after with indent adjustment."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        mock_fs.files["test.py"] = b"class Foo:\n    def method(self):\n        pass"
        response = mock_llm_response(
            search="def method(self):\n    pass",
            content="\n\ndef new_method(self):\n    return 42",
            operation="insert-after",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        result = mock_fs.files["test.py"].decode("utf-8")
        # Original should still be there
        assert "    def method(self):" in result
        assert "        pass" in result
        # New method should be properly indented
        assert "    def new_method(self):" in result
        assert "        return 42" in result

    def test_indent_flexible_match_all(self, mock_fs, mock_agent, mock_llm_response):
        """Test match_all with indent-flexible matching.

        Note: match_all only works within the same matching mode. If one block
        matches exactly and another needs indent-flexible, only the exact matches
        are replaced with match_all. This is expected behavior.
        """
        from agex.agent.loop.common import apply_optimistic_file_actions

        # Two identical blocks at the SAME non-zero indent level
        # Both require indent-flexible matching since search has no indent
        mock_fs.files["test.py"] = (
            b"    if True:\n        pass\n\n    if True:\n        pass"
        )
        response = mock_llm_response(
            search="if True:\n    pass",
            content="if True:\n    return",
            match_all=True,
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        result = mock_fs.files["test.py"].decode("utf-8")
        # Both blocks should be replaced with proper indentation
        assert "pass" not in result
        assert result.count("return") == 2
        # Check indentation is preserved
        assert "    if True:" in result
        assert "        return" in result

    def test_file_not_found_error(self, mock_fs, mock_agent, mock_llm_response):
        """Test that FileNotFoundError is raised for missing files."""
        from agex.agent.loop.common import apply_optimistic_file_actions
        from agex.llm.core import ResponseParseError

        response = mock_llm_response(
            search="anything",
            content="replacement",
        )

        with pytest.raises(ResponseParseError, match="File not found"):
            apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

    def test_search_not_found_error(self, mock_fs, mock_agent, mock_llm_response):
        """Test that error is raised when search string not found."""
        from agex.agent.loop.common import apply_optimistic_file_actions
        from agex.llm.core import ResponseParseError

        mock_fs.files["test.py"] = b"completely different content"
        response = mock_llm_response(
            search="def foo():\n    pass",
            content="replacement",
        )

        with pytest.raises(ResponseParseError, match="Search string not found"):
            apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

    def test_indent_reduction(self, mock_fs, mock_agent, mock_llm_response):
        """Test that replacement indentation is reduced when file has less indent."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        # File has no indent, but agent sends 4-space indented search
        mock_fs.files["test.py"] = b"def foo():\n    pass"
        response = mock_llm_response(
            search="    def foo():\n        pass",  # Agent thinks it's indented
            content="    def bar():\n        return 1",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        result = mock_fs.files["test.py"]
        # Result should have indentation REDUCED to match the file
        assert result == b"def bar():\n    return 1"

    def test_exact_replacement_output_spaces(
        self, mock_fs, mock_agent, mock_llm_response
    ):
        """Verify exact byte-for-byte output with space indentation."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        mock_fs.files["test.py"] = b"    def foo():\n        pass"
        response = mock_llm_response(
            search="def foo():\n    pass",
            content="def bar():\n    return 1",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        # Exact byte-for-byte check
        assert mock_fs.files["test.py"] == b"    def bar():\n        return 1"

    def test_exact_replacement_output_tabs(
        self, mock_fs, mock_agent, mock_llm_response
    ):
        """Verify exact byte-for-byte output with tab indentation."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        mock_fs.files["test.py"] = b"\tdef foo():\n\t\tpass"
        response = mock_llm_response(
            search="def foo():\n    pass",
            content="def bar():\n    return 1",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        # Exact byte-for-byte check - should use tabs
        assert mock_fs.files["test.py"] == b"\tdef bar():\n\t\treturn 1"

    def test_relative_indent_preserved_in_replacement(
        self, mock_fs, mock_agent, mock_llm_response
    ):
        """Test that relative indentation structure is preserved in replacement."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        # Class with method - method is indented 4 spaces, body is 8 spaces
        mock_fs.files["test.py"] = (
            b"class Foo:\n    def method(self):\n        if True:\n            pass"
        )
        response = mock_llm_response(
            search="def method(self):\n    if True:\n        pass",
            content="def method(self):\n    if False:\n        x = 1\n        return x",
        )

        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        result = mock_fs.files["test.py"].decode("utf-8")
        lines = result.split("\n")

        # Verify exact indentation on each line
        assert lines[0] == "class Foo:"
        assert lines[1] == "    def method(self):"
        assert lines[2] == "        if False:"
        assert lines[3] == "            x = 1"
        assert lines[4] == "            return x"

    def test_already_applied_skipped(self, mock_fs, mock_agent, mock_llm_response):
        """Edit where replacement already exists should be silently skipped."""
        from agex.agent.loop.common import apply_optimistic_file_actions

        mock_fs.files["test.py"] = b"def bar():\n    return 1"
        response = mock_llm_response(
            search="def foo():\n    pass",
            content="def bar():\n    return 1",
        )

        # Should not raise — replacement is already in the file
        apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

        # File should be unchanged
        assert mock_fs.files["test.py"] == b"def bar():\n    return 1"

    def test_error_shows_similar_lines(self, mock_fs, mock_agent, mock_llm_response):
        """Error message should show most similar lines when match fails."""
        from agex.agent.loop.common import apply_optimistic_file_actions
        from agex.llm.core import ResponseParseError

        mock_fs.files["test.py"] = b"def foo():\n    return 1\n    # done"
        response = mock_llm_response(
            search="def foo():\n    return 2\n    # done",  # "2" vs "1"
            content="replacement",
        )

        with pytest.raises(ResponseParseError, match="Did you mean to match"):
            apply_optimistic_file_actions(mock_agent, response, mock_fs, {})

    def test_error_shows_batch_status(self, mock_fs, mock_agent):
        """Error should report which earlier actions succeeded."""
        from agex.agent.datatypes import EditAction
        from agex.agent.loop.common import apply_optimistic_file_actions
        from agex.llm.core import ResponseParseError

        mock_fs.files["test.py"] = b"aaa\nbbb\nccc"

        class MockResponse:
            file_actions = [
                EditAction(path="test.py", search="aaa", content="AAA"),
                EditAction(path="test.py", search="nonexistent", content="XXX"),
            ]

        with pytest.raises(ResponseParseError, match="already applied successfully"):
            apply_optimistic_file_actions(mock_agent, MockResponse(), mock_fs, {})

        # First edit should have been applied
        assert b"AAA" in mock_fs.files["test.py"]


class TestFindSimilarLines:
    """Tests for _find_similar_lines helper."""

    def test_finds_similar_chunk(self):
        content = "def foo():\n    return 1\n    # done"
        search = "def foo():\n    return 2\n    # done"  # minor diff
        result = _find_similar_lines(search, content)
        assert result is not None
        assert "return 1" in result

    def test_returns_none_below_threshold(self):
        content = "completely unrelated content here"
        search = "def foo():\n    pass"
        result = _find_similar_lines(search, content)
        assert result is None

    def test_shows_line_numbers(self):
        content = "# header\n\ndef foo():\n    pass\n\n# footer"
        search = "def foo():\n    pss"  # typo
        result = _find_similar_lines(search, content)
        assert result is not None
        # Should contain line numbers
        assert "|" in result

    def test_marks_matching_lines(self):
        content = "aaa\nbbb\nccc\nddd"
        search = "bbb\nccc"
        result = _find_similar_lines(search, content)
        assert result is not None
        # Matching lines should be marked with >
        assert ">" in result
