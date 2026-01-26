"""
I/O commands for the terminal interpreter.
"""

import argparse
from typing import TextIO

from agex.fs.base import FileSystem
from agex.terminal.interpreter.datatypes import TerminalError


class CommandArgParser(argparse.ArgumentParser):
    """ArgumentParser that raises TerminalError instead of exiting."""

    def error(self, message):
        raise TerminalError(f"{self.prog}: {message}")

    def exit(self, status=0, message=None):
        if status != 0:
            raise TerminalError(message or "Argument parsing failed")


def echo(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Echo arguments to stdout."""
    stdout.write(" ".join(args) + "\n")


def cat(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Concatenate files and print on the standard output."""
    if not args:
        stdout.write(stdin.read())
        return

    for path in args:
        if path == "-":
            stdout.write(stdin.read())
            continue

        try:
            content_bytes = fs.read(path)
            content = content_bytes.decode("utf-8", errors="replace")
            stdout.write(content)
        except FileNotFoundError:
            raise TerminalError(f"cat: {path}: No such file or directory")
        except IsADirectoryError:
            raise TerminalError(f"cat: {path}: Is a directory")
        except Exception as e:
            raise TerminalError(f"cat: {path}: {e}")


def head(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Output the first part of files."""
    parser = CommandArgParser(prog="head", add_help=False)
    parser.add_argument("-n", "--lines", type=int, default=10)
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)
    limit = parsed.lines

    if not parsed.files:
        count = 0
        for line in stdin:
            if count >= limit:
                break
            stdout.write(line)
            count += 1
        return

    for i, path in enumerate(parsed.files):
        if len(parsed.files) > 1:
            stdout.write(f"==> {path} <==\n")

        try:
            content_bytes = fs.read(path)
            content = content_bytes.decode("utf-8", errors="replace")
            lines = content.splitlines(keepends=True)
            for line in lines[:limit]:
                stdout.write(line)

        except Exception as e:
            raise TerminalError(f"head: cannot open '{path}': {e}")

        if i < len(parsed.files) - 1:
            stdout.write("\n")


def tail(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Output the last part of files."""
    parser = CommandArgParser(prog="tail", add_help=False)
    parser.add_argument("-n", "--lines", type=int, default=10)
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)
    limit = parsed.lines

    if not parsed.files:
        lines = stdin.readlines()
        for line in lines[-limit:]:
            stdout.write(line)
        return

    for i, path in enumerate(parsed.files):
        if len(parsed.files) > 1:
            stdout.write(f"==> {path} <==\n")

        try:
            content_bytes = fs.read(path)
            content = content_bytes.decode("utf-8", errors="replace")
            lines = content.splitlines(keepends=True)
            for line in lines[-limit:]:
                stdout.write(line)

        except Exception as e:
            raise TerminalError(f"tail: cannot open '{path}': {e}")

        if i < len(parsed.files) - 1:
            stdout.write("\n")


def tee(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Read from stdin and write to stdout and files."""
    parser = CommandArgParser(prog="tee", add_help=False)
    parser.add_argument("-a", "--append", action="store_true")
    parser.add_argument("files", nargs="*")

    parsed, _ = parser.parse_known_args(args)

    content = stdin.read()

    # Write to stdout
    stdout.write(content)

    # Write to each file
    mode = "a" if parsed.append else "w"
    for path in parsed.files:
        try:
            fs.write(path, content.encode("utf-8"), mode=mode)
        except Exception as e:
            raise TerminalError(f"tee: {path}: {e}")
