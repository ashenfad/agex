"""
Diff command for the terminal interpreter.
"""

import difflib
from typing import TextIO

from agex.fs.base import FileSystem
from agex.terminal.interpreter.datatypes import TerminalError

from .io import CommandArgParser


def diff(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Compare files line by line."""
    parser = CommandArgParser(prog="diff", add_help=False)
    parser.add_argument("-u", "--unified", action="store_true")
    parser.add_argument("-c", "--context", action="store_true")
    parser.add_argument("-q", "--brief", action="store_true")
    parser.add_argument("-B", "--ignore-blank-lines", action="store_true")
    parser.add_argument("-w", "--ignore-all-space", action="store_true")
    parser.add_argument("file1", nargs="?")
    parser.add_argument("file2", nargs="?")

    parsed, _ = parser.parse_known_args(args)

    if not parsed.file1 or not parsed.file2:
        raise TerminalError(
            "diff: requires two file arguments (e.g., diff file1.txt file2.txt)"
        )

    # Read files
    def read_file(path: str) -> list[str]:
        try:
            content_bytes = fs.read(path)
            content = content_bytes.decode("utf-8", errors="replace")
            return content.splitlines(keepends=True)
        except FileNotFoundError:
            raise TerminalError(f"diff: {path}: No such file or directory")
        except IsADirectoryError:
            raise TerminalError(f"diff: {path}: Is a directory")

    file1_lines = read_file(parsed.file1)
    file2_lines = read_file(parsed.file2)

    # Apply preprocessing if needed
    def preprocess(lines: list[str]) -> list[str]:
        result = lines
        if parsed.ignore_blank_lines:
            result = [line for line in result if line.strip()]
        if parsed.ignore_all_space:
            result = [line.replace(" ", "").replace("\t", "") for line in result]
        return result

    if parsed.ignore_blank_lines or parsed.ignore_all_space:
        cmp1 = preprocess(file1_lines)
        cmp2 = preprocess(file2_lines)
    else:
        cmp1 = file1_lines
        cmp2 = file2_lines

    # Brief mode
    if parsed.brief:
        if cmp1 != cmp2:
            stdout.write(f"Files {parsed.file1} and {parsed.file2} differ\n")
        return

    # Generate diff
    if parsed.context:
        # Context format
        result = difflib.context_diff(
            file1_lines,
            file2_lines,
            fromfile=parsed.file1,
            tofile=parsed.file2,
        )
    else:
        # Unified format (default)
        result = difflib.unified_diff(
            file1_lines,
            file2_lines,
            fromfile=parsed.file1,
            tofile=parsed.file2,
        )

    for line in result:
        stdout.write(line)
