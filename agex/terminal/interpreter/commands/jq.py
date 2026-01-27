"""jq command for the terminal interpreter."""

import json
from typing import TextIO

from agex.fs.base import FileSystem
from agex.terminal.interpreter.datatypes import TerminalError
from agex.terminal.jq import JqError, ParseError, evaluate, parse_filter

from .io import CommandArgParser


def jq(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Process JSON with jq-like expressions.

    Usage:
        jq [options] <filter> [file...]

    Options:
        -r, --raw-output    Output strings without quotes
        -c, --compact       Compact output (no pretty printing)
        -e, --exit-status   Set exit status based on output
        -s, --slurp         Read all inputs into an array
        -n, --null-input    Don't read input, use null
        -j, --join-output   No newline after each output
    """
    parser = CommandArgParser(prog="jq", add_help=False)
    parser.add_argument("-r", "--raw-output", action="store_true")
    parser.add_argument("-c", "--compact", action="store_true")
    parser.add_argument("-e", "--exit-status", action="store_true")
    parser.add_argument("-s", "--slurp", action="store_true")
    parser.add_argument("-n", "--null-input", action="store_true")
    parser.add_argument("-j", "--join-output", action="store_true")
    parser.add_argument("filter", nargs="?", default=".")
    parser.add_argument("files", nargs="*")

    parsed, _ = parser.parse_known_args(args)

    # Parse the filter expression
    try:
        expr = parse_filter(parsed.filter)
    except ParseError as e:
        raise TerminalError(f"jq: parse error: {e}")

    # Collect input data
    inputs: list = []

    if parsed.null_input:
        inputs = [None]
    elif parsed.files:
        for path in parsed.files:
            try:
                content = fs.read(path).decode("utf-8")
                inputs.append(json.loads(content))
            except FileNotFoundError:
                raise TerminalError(f"jq: {path}: No such file or directory")
            except json.JSONDecodeError as e:
                raise TerminalError(f"jq: {path}: Invalid JSON: {e}")
    else:
        # Read from stdin
        content = stdin.read().strip()
        if not content:
            if parsed.null_input:
                inputs = [None]
            else:
                return  # No input, no output
        else:
            try:
                inputs.append(json.loads(content))
            except json.JSONDecodeError as e:
                raise TerminalError(f"jq: Invalid JSON from stdin: {e}")

    # Slurp mode: combine all inputs into a single array
    if parsed.slurp:
        inputs = [inputs]

    # Process each input
    for data in inputs:
        try:
            for result in evaluate(expr, data):
                _output_value(result, stdout, parsed)
        except JqError as e:
            raise TerminalError(f"jq: {e}")


def _output_value(value, stdout: TextIO, parsed) -> None:
    """Output a single value according to options."""
    if parsed.raw_output and isinstance(value, str):
        stdout.write(value)
    elif parsed.compact:
        stdout.write(json.dumps(value, separators=(",", ":")))
    else:
        stdout.write(json.dumps(value, indent=2))

    if not parsed.join_output:
        stdout.write("\n")
