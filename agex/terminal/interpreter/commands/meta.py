"""
Meta commands that invoke other commands.
"""

import io
from typing import TYPE_CHECKING, TextIO

from agex.fs.base import FileSystem
from agex.terminal.interpreter.datatypes import TerminalError

from .io import CommandArgParser

if TYPE_CHECKING:
    from agex.terminal.interpreter.datatypes import CommandFunc


def xargs(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Build and execute commands from standard input."""
    parser = CommandArgParser(prog="xargs", add_help=False)
    parser.add_argument("-I", "--replace", type=str, default=None)
    parser.add_argument("-n", "--max-args", type=int, default=None)
    parser.add_argument("-0", "--null", action="store_true")
    parser.add_argument("-t", "--verbose", action="store_true")
    parser.add_argument("command", nargs="*")

    parsed, _ = parser.parse_known_args(args)

    # Default command is echo
    if not parsed.command:
        cmd_name = "echo"
        cmd_base_args: list[str] = []
    else:
        cmd_name = parsed.command[0]
        cmd_base_args = parsed.command[1:]

    # Import BUILTINS here to avoid circular import
    from agex.terminal.interpreter.core import BUILTINS

    if cmd_name not in BUILTINS:
        raise TerminalError(f"xargs: {cmd_name}: command not found")

    cmd_func: "CommandFunc" = BUILTINS[cmd_name]

    # Read and split input
    input_text = stdin.read()
    if not input_text.strip():
        return  # No input, no commands

    if parsed.null:
        # Null-delimited
        items = [item for item in input_text.split("\0") if item]
    else:
        # Whitespace/newline delimited
        items = input_text.split()

    if not items:
        return

    def execute_cmd(cmd_args: list[str]) -> None:
        """Execute command with given args and write output to stdout."""
        if parsed.verbose:
            stdout.write(f"{cmd_name} {' '.join(cmd_args)}\n")

        cmd_stdin = io.StringIO()
        cmd_stdout = io.StringIO()
        cmd_func(cmd_args, cmd_stdin, cmd_stdout, fs)
        stdout.write(cmd_stdout.getvalue())

    if parsed.replace:
        # -I mode: run command once per item, substituting placeholder
        for item in items:
            cmd_args = [arg.replace(parsed.replace, item) for arg in cmd_base_args]
            execute_cmd(cmd_args)
    elif parsed.max_args:
        # -n mode: batch items
        for i in range(0, len(items), parsed.max_args):
            batch = items[i : i + parsed.max_args]
            execute_cmd(cmd_base_args + batch)
    else:
        # Default: all items as arguments to single command
        execute_cmd(cmd_base_args + items)
