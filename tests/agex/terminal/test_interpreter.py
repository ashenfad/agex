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


def test_simple_echo(fs):
    script = to_script("echo hello world")
    output = execute_script(script, fs)
    assert output == "hello world\n"


def test_file_operations(fs):
    # 1. mkdir
    execute_script(to_script("mkdir -p data"), fs)
    assert fs.isdir("/data")

    # 2. cd and pwd
    execute_script(to_script("cd data"), fs)
    assert fs.getcwd() == "/data"

    output = execute_script(to_script("pwd"), fs)
    assert output.strip() == "/data"

    # 3. echo redirect
    execute_script(to_script("echo 'foo bar' > test.txt"), fs)
    assert fs.read("test.txt") == b"foo bar\n"

    # 4. cat
    output = execute_script(to_script("cat test.txt"), fs)
    assert output == "foo bar\n"


def test_pipeline(fs):
    # Setup
    execute_script(to_script("echo 'line 1\nline 2\nline 3' > lines.txt"), fs)

    # Pipe: cat | head
    output = execute_script(to_script("cat lines.txt | head -n 2"), fs)
    assert output == "line 1\nline 2\n"


def test_grep_recursive(fs):
    # Setup
    execute_script(to_script("mkdir -p src"), fs)
    execute_script(to_script("echo 'def foo(): pass' > src/main.py"), fs)
    execute_script(to_script("echo 'class Bar: pass' > src/models.py"), fs)

    # Grep
    output = execute_script(to_script("grep -r 'def' src"), fs)
    assert "src/main.py:def foo(): pass" in output
    assert "src/models.py" not in output


def test_find_glob(fs):
    # Setup
    execute_script(to_script("touch a.py b.py c.txt"), fs)

    # Glob ls
    output = execute_script(to_script("ls *.py"), fs)
    assert "a.py" in output
    assert "b.py" in output
    assert "c.txt" not in output


def test_grep_file_simple(fs):
    execute_script(to_script("echo 'hello' > hello.txt"), fs)
    output = execute_script(to_script("grep 'hello' hello.txt"), fs)
    assert "hello" in output


def test_quoted_wildcard(fs):
    # Verify masking works (no glob expansion)

    # Method 1: Pipe
    # echo '*' -> prints *
    # grep -F '*' -> searches for literal *
    output = execute_script(to_script("echo '*' | grep -F '*'"), fs)
    assert "*" in output

    # Method 2: File
    execute_script(to_script("echo '*' > star.txt"), fs)
    output = execute_script(to_script("grep -F '*' star.txt"), fs)
    assert "*" in output


def test_script_error_stops_execution(fs):
    # set -e behavior
    script_text = """
    cd /nonexistent
    echo 'Should not run'
    """

    # Now that we raise TerminalError, execution stops and we can catch it
    with pytest.raises(TerminalError) as excinfo:
        execute_script(to_script(script_text), fs)

    assert "cd: no such file" in str(excinfo.value)
    # The partial output should NOT contain "Should not run"
    assert "Should not run" not in excinfo.value.partial_output


def test_grep_context_after(fs):
    """Test grep -A (after context)."""
    content = "line1\nmatch\nline3\nline4\nline5"
    execute_script(to_script(f"echo '{content}' > test.txt"), fs)

    output = execute_script(to_script("grep -A 2 'match' test.txt"), fs)
    lines = output.strip().split("\n")
    assert "match" in lines[0]
    assert "line3" in lines[1]
    assert "line4" in lines[2]


def test_grep_context_before(fs):
    """Test grep -B (before context)."""
    content = "line1\nline2\nmatch\nline4"
    execute_script(to_script(f"echo '{content}' > test.txt"), fs)

    output = execute_script(to_script("grep -B 2 'match' test.txt"), fs)
    lines = output.strip().split("\n")
    assert "line1" in lines[0]
    assert "line2" in lines[1]
    assert "match" in lines[2]


def test_grep_context_both(fs):
    """Test grep -C (before and after context)."""
    content = "line1\nline2\nmatch\nline4\nline5"
    execute_script(to_script(f"echo '{content}' > test.txt"), fs)

    output = execute_script(to_script("grep -C 1 'match' test.txt"), fs)
    lines = output.strip().split("\n")
    assert "line2" in lines[0]
    assert "match" in lines[1]
    assert "line4" in lines[2]


def test_grep_context_separator(fs):
    """Test grep context separator between non-adjacent matches."""
    content = "line1\nmatch1\nline3\nline4\nline5\nmatch2\nline7"
    execute_script(to_script(f"echo '{content}' > test.txt"), fs)

    output = execute_script(to_script("grep -C 1 'match' test.txt"), fs)
    # Should have separator between the two context groups
    assert "--" in output


def test_grep_unknown_option(fs):
    """Test that grep errors on unknown options."""
    execute_script(to_script("echo 'test' > test.txt"), fs)

    with pytest.raises(TerminalError) as excinfo:
        execute_script(to_script("grep --unknown test test.txt"), fs)
    assert "unknown option" in str(excinfo.value)


def test_grep_file_not_found(fs):
    """Test that grep gives clear error for missing files."""
    with pytest.raises(TerminalError) as excinfo:
        execute_script(to_script("grep 'test' nonexistent.txt"), fs)
    assert "No such file or directory" in str(excinfo.value)


def test_find_unknown_option(fs):
    """Test that find errors on unknown options."""
    with pytest.raises(TerminalError) as excinfo:
        execute_script(to_script("find . --unknown"), fs)
    assert "unknown option" in str(excinfo.value)


def test_ls_unknown_option(fs):
    """Test that ls errors on unknown options."""
    with pytest.raises(TerminalError) as excinfo:
        execute_script(to_script("ls --unknown"), fs)
    assert "unknown option" in str(excinfo.value)


def test_cat_unknown_option(fs):
    """Test that cat errors on unknown options."""
    execute_script(to_script("echo 'test' > test.txt"), fs)
    with pytest.raises(TerminalError) as excinfo:
        execute_script(to_script("cat --unknown test.txt"), fs)
    assert "unknown option" in str(excinfo.value)


def test_diff_unknown_option(fs):
    """Test that diff errors on unknown options."""
    execute_script(to_script("echo 'a' > a.txt"), fs)
    execute_script(to_script("echo 'b' > b.txt"), fs)
    with pytest.raises(TerminalError) as excinfo:
        execute_script(to_script("diff --unknown a.txt b.txt"), fs)
    assert "unknown option" in str(excinfo.value)
