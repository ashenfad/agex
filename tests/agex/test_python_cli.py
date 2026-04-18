"""Tests for the python script handler (agex/python_cli.py).

Exercises script execution through termish with a sandtrap sandbox,
VFS imports, stdin piping, error formatting, and task_* exclusion.
"""

import pytest
from monkeyfs import VirtualFS
from termish import execute
from termish.errors import TerminalError

from agex import Agent, clear_agent_registry
from agex.python_cli import make_python_handler


@pytest.fixture(autouse=True)
def _clean():
    clear_agent_registry()
    yield
    clear_agent_registry()


@pytest.fixture
def setup():
    """Create an agent with a VFS and a python handler."""
    agent = Agent(name="script_runner")
    # Register math so scripts can import it
    import math

    agent.module(math)

    vfs = VirtualFS()
    handler = make_python_handler(agent, vfs)
    return agent, vfs, handler


def run(vfs, handler, cmd_str):
    """Execute a termish command string with the python handler."""
    return execute(cmd_str, vfs, commands={"python": handler})


# =============================================================================
# Basic script execution
# =============================================================================


class TestBasicExecution:
    def test_run_script(self, setup):
        _, vfs, handler = setup
        vfs.write("hello.py", b"print('hello from script')\n")
        output = run(vfs, handler, "python hello.py")
        assert "hello from script" in output

    def test_run_script_with_args(self, setup):
        _, vfs, handler = setup
        vfs.write("args.py", b"import sys\nprint(' '.join(sys.argv))\n")
        output = run(vfs, handler, "python args.py foo bar")
        assert "args.py foo bar" in output

    def test_run_inline_code(self, setup):
        _, vfs, handler = setup
        output = run(vfs, handler, "python -c 'print(2 + 2)'")
        assert "4" in output

    def test_version(self, setup):
        _, vfs, handler = setup
        output = run(vfs, handler, "python --version")
        assert "Python" in output

    def test_missing_file(self, setup):
        _, vfs, handler = setup
        with pytest.raises(TerminalError, match="No such file"):
            run(vfs, handler, "python nonexistent.py")

    def test_m_flag_rejected(self, setup):
        _, vfs, handler = setup
        with pytest.raises(TerminalError, match="not supported"):
            run(vfs, handler, "python -m json")


# =============================================================================
# Fresh namespace / no task_*
# =============================================================================


class TestNamespaceIsolation:
    def test_no_task_success(self, setup):
        """task_success is not available in scripts."""
        _, vfs, handler = setup
        vfs.write("bad.py", b"task_success('done')\n")
        with pytest.raises(TerminalError, match="task_success"):
            run(vfs, handler, "python bad.py")

    def test_helpful_error_for_task_functions(self, setup):
        """Error message explains how to use task_success from REPL."""
        _, vfs, handler = setup
        vfs.write("bad.py", b"task_success('done')\n")
        with pytest.raises(TerminalError, match="<PYTHON> block"):
            run(vfs, handler, "python bad.py")

    def test_fresh_namespace_per_run(self, setup):
        """Variables from one script run don't leak to the next."""
        _, vfs, handler = setup
        vfs.write("set_var.py", b"my_var = 42\nprint(my_var)\n")
        vfs.write("read_var.py", b"print(my_var)\n")

        output = run(vfs, handler, "python set_var.py")
        assert "42" in output

        with pytest.raises(TerminalError, match="my_var"):
            run(vfs, handler, "python read_var.py")


# =============================================================================
# __name__ and __file__
# =============================================================================


class TestScriptMetadata:
    def test_name_is_main(self, setup):
        _, vfs, handler = setup
        vfs.write("check.py", b"print(__name__)\n")
        output = run(vfs, handler, "python check.py")
        assert "__main__" in output

    def test_name_guard_works(self, setup):
        """if __name__ == '__main__': block executes in script mode."""
        _, vfs, handler = setup
        vfs.write(
            "guarded.py",
            b"result = 'not main'\nif __name__ == '__main__':\n    result = 'is main'\nprint(result)\n",
        )
        output = run(vfs, handler, "python guarded.py")
        assert "is main" in output

    def test_file_is_set(self, setup):
        _, vfs, handler = setup
        vfs.write("show_file.py", b"print(__file__)\n")
        output = run(vfs, handler, "python show_file.py")
        assert "show_file.py" in output


# =============================================================================
# Module imports
# =============================================================================


class TestImports:
    def test_registered_module_available(self, setup):
        """Registered modules (math) are importable from scripts."""
        _, vfs, handler = setup
        vfs.write("use_math.py", b"import math\nprint(math.pi)\n")
        output = run(vfs, handler, "python use_math.py")
        assert "3.14" in output

    def test_vfs_module_import(self, setup):
        """Scripts can import modules written to the VFS."""
        _, vfs, handler = setup
        vfs.write("helpers/utils.py", b"def greet(name):\n    return f'hello {name}'\n")
        vfs.write(
            "main.py", b"from helpers.utils import greet\nprint(greet('world'))\n"
        )
        output = run(vfs, handler, "python main.py")
        assert "hello world" in output


# =============================================================================
# Error formatting
# =============================================================================


class TestErrors:
    def test_syntax_error(self, setup):
        _, vfs, handler = setup
        vfs.write("bad_syntax.py", b"def foo(\n")
        with pytest.raises(TerminalError):
            run(vfs, handler, "python bad_syntax.py")

    def test_runtime_error_shows_filename(self, setup):
        """Runtime errors reference the script's VFS path."""
        _, vfs, handler = setup
        vfs.write("crash.py", b"x = 1 / 0\n")
        with pytest.raises(TerminalError, match="crash.py"):
            run(vfs, handler, "python crash.py")


# =============================================================================
# Pipeline composition
# =============================================================================


class TestPipeline:
    def test_python_piped_to_grep(self, setup):
        """Script stdout can be piped to built-in commands."""
        _, vfs, handler = setup
        vfs.write(
            "gen.py",
            b"for i in range(5):\n    print(f'line {i}')\n",
        )
        output = run(vfs, handler, "python gen.py | grep 'line 3'")
        assert output.strip() == "line 3"

    def test_stdin_piped_to_python(self, setup):
        """Input can be piped into a script via stdin."""
        _, vfs, handler = setup
        vfs.write(
            "upper.py",
            b"import sys\nfor line in sys.stdin:\n    print(line.strip().upper())\n",
        )
        output = run(vfs, handler, "echo hello | python upper.py")
        assert "HELLO" in output

    def test_python_in_multi_stage_pipeline(self, setup):
        """Python script in the middle of a pipeline."""
        _, vfs, handler = setup
        vfs.write(
            "double.py",
            b"import sys\nfor line in sys.stdin:\n    n = int(line.strip())\n    print(n * 2)\n",
        )
        vfs.write("nums.txt", b"1\n2\n3\n")
        output = run(vfs, handler, "cat nums.txt | python double.py | sort -n")
        lines = output.strip().split("\n")
        assert lines == ["2", "4", "6"]
