"""
Core interpreter logic for executing terminal scripts against a FileSystem.
Functional implementation.
"""

import io
from typing import Dict, List, Optional, TextIO

from agex.fs.base import FileSystem
from agex.terminal.ast import Pipeline, Script
from agex.terminal.interpreter.datatypes import CommandFunc, TerminalError
from agex.terminal.quote_masker import mask_quotes, unmask_and_unquote

from .commands import filesystem, search
from .commands import io as io_cmds

# Static mapping of built-in commands
BUILTINS: Dict[str, CommandFunc] = {
    "pwd": filesystem.pwd,
    "cd": filesystem.cd,
    "mkdir": filesystem.mkdir,
    "ls": filesystem.ls,
    "touch": filesystem.touch,
    "cp": filesystem.cp,
    "mv": filesystem.mv,
    "rm": filesystem.rm,
    "echo": io_cmds.echo,
    "cat": io_cmds.cat,
    "head": io_cmds.head,
    "tail": io_cmds.tail,
    "grep": search.grep,
    "find": search.find,
}


def execute_script(script: Script, fs: FileSystem) -> str:
    """
    Execute a full script and return the final stdout.
    Stop on first failure (set -e).

    Args:
        script: The parsed AST.
        fs: The filesystem to operate on.

    Returns:
        Captured stdout as a string.

    Raises:
        TerminalError: If execution fails (contains partial output).
    """
    final_output = io.StringIO()

    try:
        for pipeline in script.pipelines:
            _execute_pipeline(pipeline, fs, final_output)
    except TerminalError as e:
        raise TerminalError(e.message, partial_output=final_output.getvalue()) from e
    except Exception as e:
        raise TerminalError(
            f"Unexpected error: {e}", partial_output=final_output.getvalue()
        ) from e

    return final_output.getvalue()


def _execute_pipeline(pipeline: Pipeline, fs: FileSystem, stdout: TextIO):
    """
    Execute a chain of commands.
    Raises TerminalError on failure.
    """
    if not pipeline.commands:
        return

    current_input: Optional[str] = None

    for cmd_node in pipeline.commands:
        cmd_stdin = io.StringIO(current_input) if current_input else io.StringIO()
        cmd_stdout = io.StringIO()

        # Handle Redirects (Input)
        input_redirect = next((r for r in cmd_node.redirects if r.type == "<"), None)
        if input_redirect:
            path = _resolve_path(input_redirect.target, fs)
            try:
                content_bytes = fs.read(path)
                content_str = content_bytes.decode("utf-8", errors="replace")
                cmd_stdin = io.StringIO(content_str)
            except Exception as e:
                raise TerminalError(f"{cmd_node.name}: {input_redirect.target}: {e}")

        # Prepare Args
        expanded_args = _expand_args(cmd_node.args, fs)

        # Execute Command
        if cmd_node.name in BUILTINS:
            try:
                BUILTINS[cmd_node.name](expanded_args, cmd_stdin, cmd_stdout, fs)
            except TerminalError:
                raise
            except Exception as e:
                raise TerminalError(f"{cmd_node.name}: execution error: {e}")
        else:
            raise TerminalError(f"{cmd_node.name}: command not found")

        # Capture output
        output_content = cmd_stdout.getvalue()

        # Handle Output Redirects
        output_redirects = [r for r in cmd_node.redirects if r.type in (">", ">>")]

        if output_redirects:
            for r in output_redirects:
                path = _resolve_path(r.target, fs)
                try:
                    _write_to_file(path, output_content, r.type == ">>", fs)
                except Exception as e:
                    raise TerminalError(f"{cmd_node.name}: redirect failed: {e}")
            current_input = ""
        else:
            current_input = output_content

    if current_input:
        stdout.write(current_input)


def _expand_args(args: List[str], fs: FileSystem) -> List[str]:
    """Perform globbing on arguments."""
    expanded: List[str] = []
    for arg in args:
        masked, mask_map = mask_quotes(arg)

        if ("*" in masked or "?" in masked) and not mask_map:
            try:
                matches = fs.glob(arg)
                if matches:
                    expanded.extend(matches)
                else:
                    expanded.append(arg)
            except Exception:
                expanded.append(arg)
        else:
            expanded.append(unmask_and_unquote(masked, mask_map))

    return expanded


def _resolve_path(path: str, fs: FileSystem) -> str:
    """Resolve path against CWD."""
    if path.startswith("/"):
        return path

    cwd = fs.getcwd()
    if cwd == "/":
        return f"/{path}"
    return f"{cwd}/{path}"


def _write_to_file(path: str, content: str, append: bool, fs: FileSystem):
    """Helper to write/append text to file."""
    content_bytes = content.encode("utf-8")
    mode = "a" if append else "w"
    fs.write(path, content_bytes, mode=mode)
