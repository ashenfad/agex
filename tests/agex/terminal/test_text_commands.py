"""Tests for text processing commands: wc, sort, uniq, cut, tee, diff, xargs."""

import pytest

from agex.fs.virtual import VirtualFS
from agex.state import Live
from agex.terminal.interpreter import execute_script
from agex.terminal.interpreter.datatypes import TerminalError
from agex.terminal.parser import to_script


@pytest.fixture
def fs():
    state = Live()
    return VirtualFS(state)


# =============================================================================
# wc tests
# =============================================================================


class TestWc:
    def test_line_count(self, fs):
        execute_script(to_script("echo 'line1\nline2\nline3' > test.txt"), fs)
        output = execute_script(to_script("wc -l test.txt"), fs)
        assert "3" in output
        assert "test.txt" in output

    def test_word_count(self, fs):
        execute_script(to_script("echo 'one two three four' > test.txt"), fs)
        output = execute_script(to_script("wc -w test.txt"), fs)
        assert "4" in output

    def test_byte_count(self, fs):
        execute_script(to_script("echo 'hello' > test.txt"), fs)
        output = execute_script(to_script("wc -c test.txt"), fs)
        assert "6" in output  # 'hello\n' = 6 bytes

    def test_all_counts_default(self, fs):
        execute_script(to_script("echo 'a b c' > test.txt"), fs)
        output = execute_script(to_script("wc test.txt"), fs)
        # Should show lines, words, bytes
        assert "1" in output  # 1 line
        assert "3" in output  # 3 words
        assert "test.txt" in output

    def test_stdin(self, fs):
        output = execute_script(to_script("echo 'one two three' | wc -w"), fs)
        assert "3" in output

    def test_multiple_files_total(self, fs):
        execute_script(to_script("echo 'one' > a.txt"), fs)
        execute_script(to_script("echo 'two\nthree' > b.txt"), fs)
        output = execute_script(to_script("wc -l a.txt b.txt"), fs)
        assert "a.txt" in output
        assert "b.txt" in output
        assert "total" in output

    def test_empty_file(self, fs):
        fs.write("empty.txt", b"")
        output = execute_script(to_script("wc empty.txt"), fs)
        assert "0" in output


# =============================================================================
# sort tests
# =============================================================================


class TestSort:
    def test_basic_sort(self, fs):
        execute_script(to_script("echo 'banana\napple\ncherry' > fruits.txt"), fs)
        output = execute_script(to_script("sort fruits.txt"), fs)
        lines = output.strip().split("\n")
        assert lines == ["apple", "banana", "cherry"]

    def test_reverse_sort(self, fs):
        execute_script(to_script("echo 'a\nb\nc' > abc.txt"), fs)
        output = execute_script(to_script("sort -r abc.txt"), fs)
        lines = output.strip().split("\n")
        assert lines == ["c", "b", "a"]

    def test_numeric_sort(self, fs):
        execute_script(to_script("echo '10\n2\n1\n20' > nums.txt"), fs)
        output = execute_script(to_script("sort -n nums.txt"), fs)
        lines = output.strip().split("\n")
        assert lines == ["1", "2", "10", "20"]

    def test_unique_sort(self, fs):
        execute_script(to_script("echo 'a\nb\na\nc\nb' > dups.txt"), fs)
        output = execute_script(to_script("sort -u dups.txt"), fs)
        lines = output.strip().split("\n")
        assert lines == ["a", "b", "c"]

    def test_case_insensitive(self, fs):
        execute_script(to_script("echo 'Banana\napple\nCherry' > fruits.txt"), fs)
        output = execute_script(to_script("sort -f fruits.txt"), fs)
        lines = output.strip().split("\n")
        # Case-insensitive sort, but preserves original case
        assert lines[0].lower() == "apple"

    def test_field_sort(self, fs):
        execute_script(to_script("echo 'z 1\na 3\nm 2' > data.txt"), fs)
        output = execute_script(to_script("sort -k 2 -n data.txt"), fs)
        lines = output.strip().split("\n")
        assert lines[0].startswith("z")  # 1 comes first
        assert lines[1].startswith("m")  # 2
        assert lines[2].startswith("a")  # 3

    def test_stdin_sort(self, fs):
        output = execute_script(to_script("echo 'c\na\nb' | sort"), fs)
        lines = output.strip().split("\n")
        assert lines == ["a", "b", "c"]


# =============================================================================
# uniq tests
# =============================================================================


class TestUniq:
    def test_basic_uniq(self, fs):
        execute_script(to_script("echo 'a\na\nb\nb\nb\nc' > data.txt"), fs)
        output = execute_script(to_script("uniq data.txt"), fs)
        lines = output.strip().split("\n")
        assert lines == ["a", "b", "c"]

    def test_count_uniq(self, fs):
        execute_script(to_script("echo 'a\na\nb\nc\nc\nc' > data.txt"), fs)
        output = execute_script(to_script("uniq -c data.txt"), fs)
        assert "2 a" in output or "2  a" in output.replace("  ", " ")
        assert "3 c" in output or "3  c" in output.replace("  ", " ")

    def test_duplicates_only(self, fs):
        execute_script(to_script("echo 'a\na\nb\nc\nc' > data.txt"), fs)
        output = execute_script(to_script("uniq -d data.txt"), fs)
        lines = output.strip().split("\n")
        assert "a" in lines
        assert "c" in lines
        assert "b" not in lines

    def test_unique_only(self, fs):
        execute_script(to_script("echo 'a\na\nb\nc\nc' > data.txt"), fs)
        output = execute_script(to_script("uniq -u data.txt"), fs)
        lines = output.strip().split("\n")
        assert lines == ["b"]

    def test_case_insensitive(self, fs):
        execute_script(to_script("echo 'A\na\nB' > data.txt"), fs)
        output = execute_script(to_script("uniq -i data.txt"), fs)
        lines = output.strip().split("\n")
        assert len(lines) == 2  # A/a collapse, B stays

    def test_stdin_uniq(self, fs):
        output = execute_script(to_script("echo 'x\nx\ny' | uniq"), fs)
        lines = output.strip().split("\n")
        assert lines == ["x", "y"]


# =============================================================================
# cut tests
# =============================================================================


class TestCut:
    def test_cut_fields(self, fs):
        execute_script(to_script("echo 'a\tb\tc' > data.txt"), fs)
        output = execute_script(to_script("cut -f 2 data.txt"), fs)
        assert output.strip() == "b"

    def test_cut_custom_delimiter(self, fs):
        execute_script(to_script("echo 'a,b,c' > data.txt"), fs)
        output = execute_script(to_script("cut -d ',' -f 2 data.txt"), fs)
        assert output.strip() == "b"

    def test_cut_range(self, fs):
        execute_script(to_script("echo 'a,b,c,d,e' > data.txt"), fs)
        output = execute_script(to_script("cut -d ',' -f 2-4 data.txt"), fs)
        assert output.strip() == "b,c,d"

    def test_cut_open_range(self, fs):
        execute_script(to_script("echo 'a,b,c,d,e' > data.txt"), fs)
        output = execute_script(to_script("cut -d ',' -f 3- data.txt"), fs)
        assert output.strip() == "c,d,e"

    def test_cut_characters(self, fs):
        execute_script(to_script("echo 'abcdefgh' > data.txt"), fs)
        output = execute_script(to_script("cut -c 2-4 data.txt"), fs)
        assert output.strip() == "bcd"

    def test_cut_stdin(self, fs):
        output = execute_script(to_script("echo 'x:y:z' | cut -d ':' -f 2"), fs)
        assert output.strip() == "y"

    def test_cut_multiple_fields(self, fs):
        execute_script(to_script("echo 'a,b,c,d' > data.txt"), fs)
        # Quote field spec to prevent shell from splitting on comma
        output = execute_script(to_script("cut -d ',' -f '1,3' data.txt"), fs)
        assert output.strip() == "a,c"


# =============================================================================
# tee tests
# =============================================================================


class TestTee:
    def test_tee_basic(self, fs):
        output = execute_script(to_script("echo 'hello' | tee out.txt"), fs)
        assert output.strip() == "hello"
        assert fs.read("out.txt") == b"hello\n"

    def test_tee_append(self, fs):
        execute_script(to_script("echo 'first' > out.txt"), fs)
        execute_script(to_script("echo 'second' | tee -a out.txt"), fs)
        content = fs.read("out.txt").decode()
        assert "first" in content
        assert "second" in content

    def test_tee_multiple_files(self, fs):
        execute_script(to_script("echo 'data' | tee a.txt b.txt"), fs)
        assert fs.read("a.txt") == b"data\n"
        assert fs.read("b.txt") == b"data\n"

    def test_tee_passthrough(self, fs):
        # tee should pass through to next command in pipeline
        output = execute_script(to_script("echo 'test' | tee log.txt | cat"), fs)
        assert output.strip() == "test"
        assert fs.read("log.txt") == b"test\n"


# =============================================================================
# diff tests
# =============================================================================


class TestDiff:
    def test_diff_same_files(self, fs):
        execute_script(to_script("echo 'same' > a.txt"), fs)
        execute_script(to_script("echo 'same' > b.txt"), fs)
        output = execute_script(to_script("diff a.txt b.txt"), fs)
        assert output == ""

    def test_diff_different_files(self, fs):
        execute_script(to_script("echo 'old' > a.txt"), fs)
        execute_script(to_script("echo 'new' > b.txt"), fs)
        output = execute_script(to_script("diff a.txt b.txt"), fs)
        assert "---" in output or "-old" in output
        assert "+++" in output or "+new" in output

    def test_diff_brief(self, fs):
        execute_script(to_script("echo 'x' > a.txt"), fs)
        execute_script(to_script("echo 'y' > b.txt"), fs)
        output = execute_script(to_script("diff -q a.txt b.txt"), fs)
        assert "differ" in output

    def test_diff_brief_same(self, fs):
        execute_script(to_script("echo 'same' > a.txt"), fs)
        execute_script(to_script("echo 'same' > b.txt"), fs)
        output = execute_script(to_script("diff -q a.txt b.txt"), fs)
        assert output == ""

    def test_diff_unified_format(self, fs):
        execute_script(to_script("echo 'line1\nline2' > a.txt"), fs)
        execute_script(to_script("echo 'line1\nline3' > b.txt"), fs)
        output = execute_script(to_script("diff -u a.txt b.txt"), fs)
        assert "---" in output
        assert "+++" in output
        assert "-line2" in output
        assert "+line3" in output

    def test_diff_missing_file(self, fs):
        execute_script(to_script("echo 'x' > a.txt"), fs)
        with pytest.raises(TerminalError) as exc:
            execute_script(to_script("diff a.txt missing.txt"), fs)
        assert "No such file" in str(exc.value)


# =============================================================================
# xargs tests
# =============================================================================


class TestXargs:
    def test_xargs_echo(self, fs):
        output = execute_script(to_script("echo 'a b c' | xargs echo"), fs)
        assert output.strip() == "a b c"

    def test_xargs_default_echo(self, fs):
        # Default command is echo
        output = execute_script(to_script("echo 'hello world' | xargs"), fs)
        assert output.strip() == "hello world"

    def test_xargs_with_command(self, fs):
        execute_script(to_script("echo 'test content' > test.txt"), fs)
        output = execute_script(to_script("echo 'test.txt' | xargs cat"), fs)
        assert "test content" in output

    def test_xargs_replace_mode(self, fs):
        execute_script(to_script("echo 'hello' > a.txt"), fs)
        execute_script(to_script("echo 'world' > b.txt"), fs)
        # Quote {} to prevent shell from treating braces as special
        output = execute_script(
            to_script("echo 'a.txt\nb.txt' | xargs -I '{}' cat '{}'"), fs
        )
        assert "hello" in output
        assert "world" in output

    def test_xargs_max_args(self, fs):
        # -n 1 runs command once per item
        output = execute_script(to_script("echo 'a b c' | xargs -n 1 echo"), fs)
        lines = output.strip().split("\n")
        assert len(lines) == 3
        assert lines == ["a", "b", "c"]

    def test_xargs_empty_input(self, fs):
        fs.write("empty.txt", b"")
        output = execute_script(to_script("cat empty.txt | xargs echo"), fs)
        # No input = no command executed
        assert output == ""

    def test_xargs_verbose(self, fs):
        output = execute_script(to_script("echo 'test' | xargs -t echo"), fs)
        # Should print the command being executed
        assert "echo test" in output


# =============================================================================
# cp -r and rm -r tests
# =============================================================================


class TestRecursiveCpRm:
    def test_cp_recursive(self, fs):
        # Setup directory structure
        execute_script(to_script("mkdir -p src/sub"), fs)
        execute_script(to_script("echo 'file1' > src/a.txt"), fs)
        execute_script(to_script("echo 'file2' > src/sub/b.txt"), fs)

        # Copy recursively
        execute_script(to_script("cp -r src dst"), fs)

        # Verify
        assert fs.exists("dst/a.txt")
        assert fs.exists("dst/sub/b.txt")
        assert fs.read("dst/a.txt") == b"file1\n"
        assert fs.read("dst/sub/b.txt") == b"file2\n"

    def test_cp_recursive_into_existing(self, fs):
        execute_script(to_script("mkdir -p src"), fs)
        execute_script(to_script("echo 'test' > src/file.txt"), fs)
        execute_script(to_script("mkdir -p target"), fs)

        # cp -r src target -> creates target/src/
        execute_script(to_script("cp -r src target"), fs)
        assert fs.exists("target/src/file.txt")

    def test_cp_without_r_fails(self, fs):
        execute_script(to_script("mkdir src"), fs)
        with pytest.raises(TerminalError) as exc:
            execute_script(to_script("cp src dst"), fs)
        assert "-r not specified" in str(exc.value)

    def test_rm_recursive(self, fs):
        # Setup
        execute_script(to_script("mkdir -p dir/sub"), fs)
        execute_script(to_script("echo 'x' > dir/file.txt"), fs)
        execute_script(to_script("echo 'y' > dir/sub/nested.txt"), fs)

        # Remove
        execute_script(to_script("rm -r dir"), fs)

        # Verify
        assert not fs.exists("dir")
        assert not fs.exists("dir/file.txt")
        assert not fs.exists("dir/sub")

    def test_rm_without_r_fails(self, fs):
        execute_script(to_script("mkdir mydir"), fs)
        with pytest.raises(TerminalError) as exc:
            execute_script(to_script("rm mydir"), fs)
        assert "Is a directory" in str(exc.value)

    def test_rm_force_missing(self, fs):
        # -f should not error on missing files
        output = execute_script(to_script("rm -f nonexistent.txt"), fs)
        assert output == ""

    def test_rm_force_without_f_errors(self, fs):
        with pytest.raises(TerminalError) as exc:
            execute_script(to_script("rm nonexistent.txt"), fs)
        assert "No such file" in str(exc.value)


# =============================================================================
# Pipeline integration tests
# =============================================================================


class TestPipelineIntegration:
    def test_sort_uniq_pipeline(self, fs):
        execute_script(to_script("echo 'b\na\nb\nc\na' > data.txt"), fs)
        output = execute_script(to_script("cat data.txt | sort | uniq"), fs)
        lines = output.strip().split("\n")
        assert lines == ["a", "b", "c"]

    def test_grep_wc_pipeline(self, fs):
        execute_script(
            to_script("echo 'error: foo\ninfo: bar\nerror: baz' > log.txt"), fs
        )
        output = execute_script(to_script("grep error log.txt | wc -l"), fs)
        assert "2" in output

    def test_cut_sort_uniq_pipeline(self, fs):
        execute_script(
            to_script("echo 'user1,login\nuser2,logout\nuser1,login' > events.txt"), fs
        )
        output = execute_script(
            to_script("cat events.txt | cut -d ',' -f 1 | sort | uniq"), fs
        )
        lines = output.strip().split("\n")
        assert lines == ["user1", "user2"]

    def test_find_xargs_grep(self, fs):
        execute_script(to_script("mkdir -p src"), fs)
        execute_script(to_script("echo 'TODO: fix this' > src/main.py"), fs)
        execute_script(to_script("echo 'nothing here' > src/other.py"), fs)

        output = execute_script(
            to_script("find src -name '*.py' | xargs grep TODO"), fs
        )
        assert "TODO" in output
        assert "main.py" in output
