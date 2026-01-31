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
    parser.add_argument("-A", "--after-context", type=int, default=0)
    parser.add_argument("-B", "--before-context", type=int, default=0)
    parser.add_argument("-C", "--context", type=int, default=0)
    parser.add_argument("pattern")
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"grep: unknown option: {unknown[0]}")

    # -C sets both before and after context
    before_context = parsed.before_context
    after_context = parsed.after_context
    if parsed.context > 0:
        before_context = max(before_context, parsed.context)
        after_context = max(after_context, parsed.context)

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
    has_context = before_context > 0 or after_context > 0

    def process_content(content: str, label: str | None) -> None:
        nonlocal matches_total
        lines = content.splitlines()

        if has_context:
            # Context mode: need to track which lines to print
            matching_lines: set[int] = set()
            context_lines: set[int] = set()

            # First pass: find all matching lines
            for i, line in enumerate(lines):
                match = regex.search(line)
                is_match = bool(match)
                if parsed.invert_match:
                    is_match = not is_match
                if is_match:
                    matching_lines.add(i)
                    matches_total += 1
                    if parsed.files_with_matches:
                        if label:
                            stdout.write(f"{label}\n")
                        return

            # Second pass: mark context lines
            for match_idx in matching_lines:
                for ctx_idx in range(
                    max(0, match_idx - before_context),
                    min(len(lines), match_idx + after_context + 1),
                ):
                    if ctx_idx not in matching_lines:
                        context_lines.add(ctx_idx)

            # Third pass: output lines in order with separators
            lines_to_print = sorted(matching_lines | context_lines)
            prev_idx = -2  # Track for separator insertion

            for idx in lines_to_print:
                # Print separator if there's a gap
                if prev_idx >= 0 and idx > prev_idx + 1:
                    stdout.write("--\n")
                prev_idx = idx

                line = lines[idx]
                is_match = idx in matching_lines
                separator = ":" if is_match else "-"

                prefix = ""
                if label:
                    prefix += f"{label}{separator}"
                if parsed.line_number:
                    prefix += f"{idx+1}{separator}"

                if prefix:
                    stdout.write(f"{prefix}{line}\n")
                else:
                    stdout.write(f"{line}\n")
        else:
            # Non-context mode: original behavior
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

        except FileNotFoundError:
            raise TerminalError(f"grep: {filepath}: No such file or directory")
        except IsADirectoryError:
            raise TerminalError(f"grep: {filepath}: Is a directory")
        except Exception as e:
            raise TerminalError(f"grep: {filepath}: {e}")


def find(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Search for files in a directory hierarchy."""
    parser = CommandArgParser(prog="find", add_help=False)
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("-name")
    parser.add_argument("-type", choices=["f", "d"])

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"find: unknown option: {unknown[0]}")

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
