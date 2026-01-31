"""
Text processing commands for the terminal interpreter.
"""

from typing import TextIO

from agex.fs.base import FileSystem
from agex.terminal.interpreter.datatypes import TerminalError

from .io import CommandArgParser


def wc(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Word, line, character, and byte count."""
    parser = CommandArgParser(prog="wc", add_help=False)
    parser.add_argument("-l", "--lines", action="store_true")
    parser.add_argument("-w", "--words", action="store_true")
    parser.add_argument("-c", "--bytes", action="store_true")
    parser.add_argument("-m", "--chars", action="store_true")
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"wc: unknown option: {unknown[0]}")

    # If no flags specified, show all three (lines, words, bytes)
    show_lines = parsed.lines
    show_words = parsed.words
    show_bytes = parsed.bytes or parsed.chars  # -m same as -c for UTF-8
    if not (show_lines or show_words or show_bytes):
        show_lines = show_words = show_bytes = True

    totals = {"lines": 0, "words": 0, "bytes": 0}
    results: list[tuple[dict[str, int], str]] = []

    def count_content(content: str, name: str):
        counts = {
            "lines": content.count("\n"),
            "words": len(content.split()),
            "bytes": len(content.encode("utf-8")),
        }
        results.append((counts, name))
        for key in totals:
            totals[key] += counts[key]

    if not parsed.files:
        # Read from stdin
        content = stdin.read()
        count_content(content, "")
    else:
        for path in parsed.files:
            try:
                content_bytes = fs.read(path)
                content = content_bytes.decode("utf-8", errors="replace")
                count_content(content, path)
            except FileNotFoundError:
                raise TerminalError(f"wc: {path}: No such file or directory")
            except IsADirectoryError:
                raise TerminalError(f"wc: {path}: Is a directory")

    # Calculate width for formatting (based on largest number)
    max_val = max(totals.values()) if totals["bytes"] > 0 else 1
    width = len(str(max_val))

    def format_line(counts: dict[str, int], name: str) -> str:
        parts = []
        if show_lines:
            parts.append(f"{counts['lines']:>{width}}")
        if show_words:
            parts.append(f"{counts['words']:>{width}}")
        if show_bytes:
            parts.append(f"{counts['bytes']:>{width}}")
        line = " ".join(parts)
        if name:
            line += f" {name}"
        return line

    for counts, name in results:
        stdout.write(format_line(counts, name) + "\n")

    # Show total if multiple files
    if len(results) > 1:
        stdout.write(format_line(totals, "total") + "\n")


def sort(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Sort lines of text."""
    parser = CommandArgParser(prog="sort", add_help=False)
    parser.add_argument("-r", "--reverse", action="store_true")
    parser.add_argument("-n", "--numeric-sort", action="store_true")
    parser.add_argument("-u", "--unique", action="store_true")
    parser.add_argument("-f", "--ignore-case", action="store_true")
    parser.add_argument("-k", "--key", type=str, default=None)
    parser.add_argument("-t", "--field-separator", type=str, default=None)
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"sort: unknown option: {unknown[0]}")

    # Collect all lines
    lines: list[str] = []
    if not parsed.files:
        lines = stdin.read().splitlines()
    else:
        for path in parsed.files:
            try:
                content_bytes = fs.read(path)
                content = content_bytes.decode("utf-8", errors="replace")
                lines.extend(content.splitlines())
            except FileNotFoundError:
                raise TerminalError(f"sort: {path}: No such file or directory")
            except IsADirectoryError:
                raise TerminalError(f"sort: {path}: Is a directory")

    # Parse -k field number (1-indexed)
    field_num = None
    if parsed.key:
        try:
            field_num = int(parsed.key.split(",")[0].split(".")[0])
        except ValueError:
            raise TerminalError(f"sort: invalid field specification: {parsed.key}")

    def make_key(line: str):
        val = line
        if field_num is not None:
            if parsed.field_separator:
                fields = line.split(parsed.field_separator)
            else:
                fields = line.split()
            if field_num <= len(fields):
                val = fields[field_num - 1]
            else:
                val = ""

        if parsed.ignore_case:
            val = val.lower()

        if parsed.numeric_sort:
            try:
                return (0, float(val))
            except ValueError:
                # Non-numeric sorts after numeric in GNU sort
                return (1, val)
        return val

    # Sort with stable sort
    sorted_lines = sorted(lines, key=make_key, reverse=parsed.reverse)

    # Deduplicate if -u (preserve first occurrence based on sort key)
    if parsed.unique:
        seen_keys: set = set()
        unique_lines: list[str] = []
        for line in sorted_lines:
            key = make_key(line)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_lines.append(line)
        sorted_lines = unique_lines

    for line in sorted_lines:
        stdout.write(line + "\n")


def uniq(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Report or omit repeated lines."""
    parser = CommandArgParser(prog="uniq", add_help=False)
    parser.add_argument("-c", "--count", action="store_true")
    parser.add_argument("-d", "--repeated", action="store_true")
    parser.add_argument("-u", "--unique", action="store_true")
    parser.add_argument("-i", "--ignore-case", action="store_true")
    parser.add_argument("file", nargs="?")

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"uniq: unknown option: {unknown[0]}")

    # Read input
    if parsed.file:
        try:
            content_bytes = fs.read(parsed.file)
            content = content_bytes.decode("utf-8", errors="replace")
            lines = content.splitlines()
        except FileNotFoundError:
            raise TerminalError(f"uniq: {parsed.file}: No such file or directory")
        except IsADirectoryError:
            raise TerminalError(f"uniq: {parsed.file}: Is a directory")
    else:
        lines = stdin.read().splitlines()

    if not lines:
        return

    def compare_key(line: str) -> str:
        return line.lower() if parsed.ignore_case else line

    # Group adjacent identical lines
    groups: list[tuple[int, str]] = []  # (count, original_line)
    current_line = lines[0]
    current_key = compare_key(current_line)
    count = 1

    for line in lines[1:]:
        key = compare_key(line)
        if key == current_key:
            count += 1
        else:
            groups.append((count, current_line))
            current_line = line
            current_key = key
            count = 1
    groups.append((count, current_line))

    # Output based on flags
    for count, line in groups:
        # -d: only print duplicates (count > 1)
        if parsed.repeated and count == 1:
            continue
        # -u: only print unique (count == 1)
        if parsed.unique and count > 1:
            continue

        if parsed.count:
            stdout.write(f"{count:7d} {line}\n")
        else:
            stdout.write(line + "\n")


def cut(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Remove sections from each line of files."""
    parser = CommandArgParser(prog="cut", add_help=False)
    parser.add_argument("-d", "--delimiter", type=str, default="\t")
    parser.add_argument("-f", "--fields", type=str, default=None)
    parser.add_argument("-c", "--characters", type=str, default=None)
    parser.add_argument("-b", "--bytes", type=str, default=None)
    parser.add_argument("--complement", action="store_true")
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"cut: unknown option: {unknown[0]}")

    # Must specify one of -f, -c, or -b
    if not (parsed.fields or parsed.characters or parsed.bytes):
        raise TerminalError(
            "cut: you must specify -f (fields), -c (characters), or -b (bytes) "
            "(e.g., cut -d ',' -f 1 file.txt)"
        )

    def parse_ranges(spec: str) -> list[tuple[int | None, int | None]]:
        """Parse field/char specification like '1,3-5,7-' into ranges."""
        ranges: list[tuple[int | None, int | None]] = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                start = int(start_str) if start_str else 1
                end = int(end_str) if end_str else None
                ranges.append((start, end))
            else:
                n = int(part)
                ranges.append((n, n))
        return ranges

    def select_items(
        items: list[str], ranges: list[tuple[int | None, int | None]], complement: bool
    ) -> list[str]:
        """Select items based on 1-indexed ranges."""
        if complement:
            # Select all indices NOT in ranges
            excluded: set[int] = set()
            for start, end in ranges:
                if end is None:
                    end = len(items)
                for i in range(start, end + 1):
                    excluded.add(i)
            return [items[i] for i in range(len(items)) if (i + 1) not in excluded]
        else:
            result: list[str] = []
            for start, end in ranges:
                if end is None:
                    result.extend(items[start - 1 :])
                else:
                    result.extend(items[start - 1 : end])
            return result

    # Parse the ranges
    if parsed.fields:
        try:
            ranges = parse_ranges(parsed.fields)
        except ValueError:
            raise TerminalError(f"cut: invalid field specification: {parsed.fields}")
        mode = "fields"
    elif parsed.characters:
        try:
            ranges = parse_ranges(parsed.characters)
        except ValueError:
            raise TerminalError(
                f"cut: invalid character specification: {parsed.characters}"
            )
        mode = "chars"
    else:  # bytes
        try:
            ranges = parse_ranges(parsed.bytes)
        except ValueError:
            raise TerminalError(f"cut: invalid byte specification: {parsed.bytes}")
        mode = "chars"  # Same as chars for our purposes

    # Collect lines
    lines: list[str] = []
    if not parsed.files:
        lines = stdin.read().splitlines()
    else:
        for path in parsed.files:
            try:
                content_bytes = fs.read(path)
                content = content_bytes.decode("utf-8", errors="replace")
                lines.extend(content.splitlines())
            except FileNotFoundError:
                raise TerminalError(f"cut: {path}: No such file or directory")
            except IsADirectoryError:
                raise TerminalError(f"cut: {path}: Is a directory")

    # Process each line
    for line in lines:
        if mode == "fields":
            fields = line.split(parsed.delimiter)
            selected = select_items(fields, ranges, parsed.complement)
            stdout.write(parsed.delimiter.join(selected) + "\n")
        else:  # chars/bytes
            chars = list(line)
            selected = select_items(chars, ranges, parsed.complement)
            stdout.write("".join(selected) + "\n")
