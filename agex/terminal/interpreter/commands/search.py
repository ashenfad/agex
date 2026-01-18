"Search commands (grep, find) for the terminal interpreter."

import fnmatch
import re
from typing import TextIO

from agex.fs.base import FileSystem
from agex.terminal.interpreter.commands.io import CommandArgParser
from agex.terminal.interpreter.datatypes import TerminalError


def grep(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Print lines that match patterns."""
    parser = CommandArgParser(prog="grep", add_help=False)
    parser.add_argument("-i", "--ignore-case", action="store_true")
    parser.add_argument("-n", "--line-number", action="store_true")
    parser.add_argument("-r", "-R", "--recursive", action="store_true")
    parser.add_argument("-l", "--files-with-matches", action="store_true")
    parser.add_argument("-v", "--invert-match", action="store_true")
    parser.add_argument("-F", "--fixed-strings", action="store_true")
    parser.add_argument("-E", "--extended-regexp", action="store_true")
    parser.add_argument("pattern")
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)

    flags = 0
    if parsed.ignore_case:
        flags |= re.IGNORECASE

    pattern_str = parsed.pattern
    if parsed.fixed_strings:
        pattern_str = re.escape(pattern_str)

    try:
        regex = re.compile(pattern_str, flags)
    except re.error as e:
        raise TerminalError(f"grep: invalid regex: {e}")

    matches_total = 0

    def process_content(content: str, label: str | None) -> None:
        nonlocal matches_total
        lines = content.splitlines()

        for i, line in enumerate(lines):
            match = regex.search(line)
            is_match = bool(match)

            if parsed.invert_match:
                is_match = not is_match

            if is_match:
                matches_total += 1
                if parsed.files_with_matches:
                    if label:
                        stdout.write(f"{label}\n")
                    return  # Stop processing this file

                prefix = ""
                if label:
                    prefix += f"{label}:"
                if parsed.line_number:
                    prefix += f"{i+1}:"

                if prefix:
                    stdout.write(f"{prefix}{line}\n")
                else:
                    stdout.write(f"{line}\n")

    if not parsed.files and not parsed.recursive:
        content = stdin.read()
        process_content(content, None)
        return

    files_to_search = []

    if not parsed.files:
        if parsed.recursive:
            root = "."
            try:
                all_files = fs.list_detailed(root, recursive=True)
                for f in all_files:
                    if not f.is_dir:
                        files_to_search.append(f.path)
            except Exception as e:
                raise TerminalError(f"grep: {e}")
    else:
        for path in parsed.files:
            if fs.isdir(path):
                if parsed.recursive:
                    try:
                        all_files = fs.list_detailed(path, recursive=True)
                        for f in all_files:
                            if not f.is_dir:
                                files_to_search.append(f.path)
                    except Exception as e:
                        raise TerminalError(f"grep: {path}: {e}")
                else:
                    raise TerminalError(f"grep: {path}: Is a directory")
            else:
                files_to_search.append(path)

    multiple_files = len(files_to_search) > 1 or parsed.recursive

    for filepath in files_to_search:
        try:
            content_bytes = fs.read(filepath)
            content = content_bytes.decode("utf-8", errors="replace")

            label = filepath if (multiple_files or parsed.recursive) else None
            process_content(content, label)

        except Exception as e:
            raise TerminalError(f"grep: {filepath}: {e}")


def find(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Search for files in a directory hierarchy."""
    parser = CommandArgParser(prog="find", add_help=False)
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("-name")
    parser.add_argument("-type", choices=["f", "d"])

    parsed, unknown = parser.parse_known_args(args)

    root_path = parsed.path
    try:
        all_items = fs.list_detailed(root_path, recursive=True)

        for item in all_items:
            if parsed.type == "f" and item.is_dir:
                continue
            if parsed.type == "d" and not item.is_dir:
                continue

            if parsed.name:
                if not fnmatch.fnmatch(item.name, parsed.name):
                    continue

            stdout.write(f"{item.path}\n")

    except Exception as e:
        raise TerminalError(f"find: {e}")
