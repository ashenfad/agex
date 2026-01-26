import pytest

from agex.terminal.parser import ParseError, to_script


def test_empty_string():
    script = to_script("")
    assert script.pipelines == []

    script = to_script("   ")
    assert script.pipelines == []


def test_single_command():
    script = to_script("ls -la")
    assert len(script.pipelines) == 1
    pipeline = script.pipelines[0]
    assert len(pipeline.commands) == 1

    cmd = pipeline.commands[0]
    assert cmd.name == "ls"
    assert cmd.args == ["-la"]
    assert cmd.redirects == []


def test_command_with_quotes():
    # to_script preserves quotes now!
    script = to_script('grep "search term" file.txt')
    cmd = script.pipelines[0].commands[0]

    assert cmd.name == "grep"
    assert cmd.args == ['"search term"', "file.txt"]


def test_pipeline():
    script = to_script("cat file.txt | grep error")
    assert len(script.pipelines) == 1
    pipeline = script.pipelines[0]
    assert len(pipeline.commands) == 2

    cmd1 = pipeline.commands[0]
    assert cmd1.name == "cat"
    assert cmd1.args == ["file.txt"]

    cmd2 = pipeline.commands[1]
    assert cmd2.name == "grep"
    assert cmd2.args == ["error"]


def test_redirects():
    # Output overwrite
    script = to_script("echo hello > out.txt")
    cmd = script.pipelines[0].commands[0]
    assert len(cmd.redirects) == 1
    assert cmd.redirects[0].type == ">"
    assert cmd.redirects[0].target == "out.txt"
    assert cmd.args == ["hello"]  # 'hello' is an arg, > out.txt is redirect

    # Append
    script = to_script("log >> app.log")
    cmd = script.pipelines[0].commands[0]
    assert cmd.redirects[0].type == ">>"
    assert cmd.redirects[0].target == "app.log"

    # Input
    script = to_script("cat < input.txt")
    cmd = script.pipelines[0].commands[0]
    assert cmd.redirects[0].type == "<"
    assert cmd.redirects[0].target == "input.txt"


def test_multiple_redirects():
    # order should differ based on parse, but logically valid
    script = to_script("cat < input.txt > output.txt")
    cmd = script.pipelines[0].commands[0]

    assert len(cmd.redirects) == 2
    types = {r.type for r in cmd.redirects}
    assert types == {"<", ">"}


def test_redirect_location_independence():
    # Redirects can appear anywhere
    script = to_script("> out.txt echo hello")
    cmd = script.pipelines[0].commands[0]

    assert cmd.name == "echo"
    assert cmd.args == ["hello"]
    assert cmd.redirects[0].target == "out.txt"


def test_multiple_pipelines_semicolon():
    script = to_script("cd /tmp; ls")
    assert len(script.pipelines) == 2

    assert script.pipelines[0].commands[0].name == "cd"
    assert script.pipelines[1].commands[0].name == "ls"


def test_punctuation_no_spaces():
    # The parser should handle operators without surrounding spaces
    script = to_script("ls|grep py>out.txt")

    pipeline = script.pipelines[0]
    assert len(pipeline.commands) == 2

    cmd1 = pipeline.commands[0]
    assert cmd1.name == "ls"

    cmd2 = pipeline.commands[1]
    assert cmd2.name == "grep"
    assert cmd2.args == ["py"]
    assert len(cmd2.redirects) == 1
    assert cmd2.redirects[0].target == "out.txt"


def test_errors():
    with pytest.raises(ParseError, match="Expected filename"):
        to_script("ls >")

    with pytest.raises(ParseError, match="Expected filename"):
        to_script("ls > | grep")

    with pytest.raises(ParseError, match="Unexpected pipe"):
        to_script("| ls")


# --- Complex Scripts Tests ---


def test_multiline_setup_script():
    """Test a typical setup script with multiple commands on newlines."""
    # Now using actual newlines, which our parser should handle correctly
    script_text = """
    mkdir -p tests/data
    cd tests/data
    echo "test data" > sample.txt
    ls -la
    """
    script = to_script(script_text)

    assert len(script.pipelines) == 4

    # 1. mkdir
    cmd1 = script.pipelines[0].commands[0]
    assert cmd1.name == "mkdir"
    assert cmd1.args == ["-p", "tests/data"]

    # 2. cd
    cmd2 = script.pipelines[1].commands[0]
    assert cmd2.name == "cd"
    assert cmd2.args == ["tests/data"]

    # 3. echo
    cmd3 = script.pipelines[2].commands[0]
    assert cmd3.name == "echo"
    assert cmd3.args == ['"test data"']  # Quoted!
    assert cmd3.redirects[0].target == "sample.txt"
    assert cmd3.redirects[0].type == ">"

    # 4. ls
    cmd4 = script.pipelines[3].commands[0]
    assert cmd4.name == "ls"
    assert cmd4.args == ["-la"]


def test_newline_inside_quotes():
    """Test that newlines inside quotes are NOT treated as separators."""
    script_text = """
    echo "Line 1
Line 2"
    ls
    """
    script = to_script(script_text)

    assert len(script.pipelines) == 2

    # The echo command should have one argument with an embedded newline
    cmd1 = script.pipelines[0].commands[0]
    assert cmd1.name == "echo"
    assert len(cmd1.args) == 1
    assert cmd1.args[0] == '"Line 1\nLine 2"'  # Quoted!

    # The ls command is separate
    cmd2 = script.pipelines[1].commands[0]
    assert cmd2.name == "ls"


def test_complex_grep_regex():
    """Test complex arguments with quotes and regex patterns."""
    # Note: single quotes in shell string for regex to avoid escaping issues
    input_text = r"grep -rE '^class\s+.*:' ."

    script = to_script(input_text)
    assert len(script.pipelines) == 1
    cmd = script.pipelines[0].commands[0]

    assert cmd.name == "grep"
    # Ensure the regex was preserved as a single argument including spaces
    # It was single quoted in input, so it should be single quoted in args
    assert cmd.args == ["-rE", r"'^class\s+.*:'", "."]


def test_long_pipeline_log_analysis():
    """Test a 3-stage pipeline typical of log analysis."""
    # cat logs | grep ERROR | tail -n 5 > recent_errors.txt
    input_text = (
        "cat /var/log/syslog | grep 'ERROR 500' | tail -n 5 > recent_errors.txt"
    )

    script = to_script(input_text)
    assert len(script.pipelines) == 1
    pipeline = script.pipelines[0]
    assert len(pipeline.commands) == 3

    # 1. cat
    assert pipeline.commands[0].name == "cat"
    assert pipeline.commands[0].args == ["/var/log/syslog"]

    # 2. grep
    assert pipeline.commands[1].name == "grep"
    assert pipeline.commands[1].args == ["'ERROR 500'"]  # Quoted

    # 3. tail
    last_cmd = pipeline.commands[2]
    assert last_cmd.name == "tail"
    assert last_cmd.args == ["-n", "5"]
    assert len(last_cmd.redirects) == 1
    assert last_cmd.redirects[0].target == "recent_errors.txt"
    assert last_cmd.redirects[0].type == ">"


def test_find_xargs_simulation():
    """
    Test parsing of a find command.
    Even though we don't interpret xargs logic here, the parser should handle it as a command.
    """
    input_text = 'find src -name "*.py" -type f'

    script = to_script(input_text)
    cmd = script.pipelines[0].commands[0]

    assert cmd.name == "find"
    assert cmd.args == ["src", "-name", '"*.py"', "-type", "f"]  # Quoted


def test_mixed_quotes_and_escapes():
    """Test handling of mixed quotes and escaped characters."""
    # echo "It's a me, Mario!" 'And "Luigi"'
    input_text = 'echo "It\'s a me, Mario!" \'And "Luigi"\''

    script = to_script(input_text)
    cmd = script.pipelines[0].commands[0]

    assert cmd.name == "echo"
    assert len(cmd.args) == 2
    # The parser preserved them as separate tokens because they were separate quoted blocks in mask?
    # Wait, shlex splits them if they are separate.
    # Input has space between them? Yes.
    assert cmd.args[0] == '"It\'s a me, Mario!"'
    assert cmd.args[1] == "'And \"Luigi\"'"


def test_implicit_concatenation_and_redirects():
    """Test standard shell redirect location flexibility in complex chain."""
    # < input.txt grep "foo" > output.txt
    input_text = "< input.txt grep 'foo' > output.txt"

    script = to_script(input_text)
    cmd = script.pipelines[0].commands[0]

    assert cmd.name == "grep"
    assert cmd.args == ["'foo'"]  # Quoted

    targets = {r.target for r in cmd.redirects}
    assert targets == {"input.txt", "output.txt"}

    types = {r.type for r in cmd.redirects}
    assert types == {"<", ">"}


def test_line_continuation():
    """Test backslash-newline line continuation."""
    # Simple continuation
    script = to_script("echo hello \\\nworld")
    cmd = script.pipelines[0].commands[0]
    assert cmd.name == "echo"
    assert cmd.args == ["hello", "world"]

    # Multiple continuations with indentation
    script = to_script("git add \\\n  file1.txt \\\n  file2.txt")
    cmd = script.pipelines[0].commands[0]
    assert cmd.name == "git"
    assert cmd.args == ["add", "file1.txt", "file2.txt"]

    # Continuation in pipeline
    script = to_script("cat file.txt | \\\n  grep pattern | \\\n  wc -l")
    assert len(script.pipelines) == 1
    assert len(script.pipelines[0].commands) == 3

    # Quoted backslash-n should NOT be treated as continuation
    script = to_script(r"echo 'hello\nworld'")
    cmd = script.pipelines[0].commands[0]
    assert cmd.args == [r"'hello\nworld'"]
    assert "\n" not in cmd.args[0]  # No actual newline
