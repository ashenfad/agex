"""
Filesystem commands for the terminal interpreter.
"""

import argparse
from typing import List, TextIO

from agex.fs.base import FileSystem
from agex.terminal.interpreter.datatypes import TerminalError


class CommandArgParser(argparse.ArgumentParser):
    """ArgumentParser that raises TerminalError instead of exiting."""

    def error(self, message):
        raise TerminalError(f"{self.prog}: {message}")

    def exit(self, status=0, message=None):
        if status != 0:
            raise TerminalError(message or "Argument parsing failed")


def pwd(args: List[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Print working directory."""
    stdout.write(fs.getcwd() + "\n")


def cd(args: List[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Change directory."""
    if not args:
        path = "/"
    else:
        path = args[0]

    try:
        fs.chdir(path)
    except FileNotFoundError:
        raise TerminalError(f"cd: no such file or directory: {path}")
    except NotADirectoryError:
        raise TerminalError(f"cd: not a directory: {path}")
    except Exception as e:
        raise TerminalError(f"cd: {e}")


def mkdir(args: List[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Make directories."""
    parser = CommandArgParser(prog="mkdir", add_help=False)
    parser.add_argument("-p", "--parents", action="store_true")
    parser.add_argument("paths", nargs="+")

    # We don't catch parser errors here, let them propagate as TerminalError
    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"mkdir: unknown arguments: {unknown}")

    for path in parsed.paths:
        try:
            if parsed.parents:
                fs.makedirs(path, exist_ok=True)
            else:
                fs.mkdir(path, exist_ok=False)
        except FileExistsError:
            raise TerminalError(f"mkdir: cannot create directory '{path}': File exists")
        except Exception as e:
            raise TerminalError(f"mkdir: cannot create directory '{path}': {e}")


def ls(args: List[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """List directory contents."""
    parser = CommandArgParser(prog="ls", add_help=False)
    parser.add_argument("-l", action="store_true")
    parser.add_argument("-a", action="store_true")
    parser.add_argument("-R", action="store_true")
    parser.add_argument("paths", nargs="*", default=["."])

    parsed, unknown = parser.parse_known_args(args)

    for i, path in enumerate(parsed.paths):
        if len(parsed.paths) > 1:
            stdout.write(f"{path}:\n")

        try:
            # Check if it is a file first
            if fs.isfile(path):
                if parsed.l:
                    # List detailed for file
                    meta = fs.stat(path)
                    size = str(meta.size).rjust(8)
                    time = (
                        meta.modified_at[:16].replace("T", " ")
                        if meta.modified_at
                        else " " * 16
                    )
                    stdout.write(f"-rw-r--r-- 1 agent agent {size} {time} {path}\n")
                else:
                    stdout.write(f"{path}\n")
                continue

            if parsed.l:
                items = fs.list_detailed(path, recursive=parsed.R)
                for item in items:
                    if not parsed.a and item.name.startswith("."):
                        continue
                    type_char = "d" if item.is_dir else "-"
                    size = str(item.size).rjust(8)
                    time = (
                        item.modified_at[:16].replace("T", " ")
                        if item.modified_at
                        else " " * 16
                    )
                    stdout.write(
                        f"{type_char}rw-r--r-- 1 agent agent {size} {time} {item.path}\n"
                    )
            else:
                items_str = fs.list(path, recursive=parsed.R)
                filtered = [
                    p
                    for p in items_str
                    if parsed.a or not p.split("/")[-1].startswith(".")
                ]
                if filtered:
                    stdout.write("\n".join(filtered) + "\n")

        except FileNotFoundError:
            raise TerminalError(
                f"ls: cannot access '{path}': No such file or directory"
            )
        except NotADirectoryError:
            # Should be handled by isfile check above, but if it raced or failed:
            raise TerminalError(f"ls: cannot access '{path}': Not a directory")
        except Exception as e:
            raise TerminalError(f"ls: {e}")

        if i < len(parsed.paths) - 1:
            stdout.write("\n")


def touch(args: List[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Update timestamps or create empty files."""
    if not args:
        raise TerminalError("touch: missing file operand")

    for path in args:
        try:
            if not fs.exists(path):
                fs.write(path, b"")
            else:
                content = fs.read(path)
                fs.write(path, content)
        except Exception as e:
            raise TerminalError(f"touch: {e}")


def cp(args: List[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Copy files."""
    parser = CommandArgParser(prog="cp", add_help=False)
    parser.add_argument("-r", "-R", action="store_true")
    parser.add_argument("src", nargs="+")
    parser.add_argument("dst")

    parsed, unknown = parser.parse_known_args(args)
    if len(parsed.src) > 1:
        sources = parsed.src
        dst = parsed.dst
        if not fs.isdir(dst):
            raise TerminalError(f"cp: target '{dst}' is not a directory")
    else:
        sources = [parsed.src[0]]
        dst = parsed.dst

    for src in sources:
        try:
            if fs.isdir(src):
                if not parsed.r:
                    raise TerminalError(
                        f"cp: -r not specified; omitting directory '{src}'"
                    )

                raise TerminalError("cp: recursive copy not fully implemented yet")

            content = fs.read(src)

            if fs.isdir(dst):
                import os.path

                filename = os.path.basename(src)
                target_path = f"{dst}/{filename}"
            else:
                target_path = dst

            fs.write(target_path, content)

        except FileNotFoundError:
            raise TerminalError(f"cp: cannot stat '{src}': No such file or directory")
        except TerminalError:
            raise
        except Exception as e:
            raise TerminalError(f"cp: {e}")


def mv(args: List[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Move files."""
    if len(args) != 2:
        raise TerminalError("mv: missing destination file operand")

    src, dst = args
    try:
        fs.rename(src, dst)
    except FileNotFoundError:
        raise TerminalError(f"mv: cannot stat '{src}': No such file or directory")
    except Exception as e:
        raise TerminalError(f"mv: {e}")


def rm(args: List[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Remove files."""
    parser = CommandArgParser(prog="rm", add_help=False)
    parser.add_argument("-r", "-R", action="store_true")
    parser.add_argument("paths", nargs="+")

    parsed, unknown = parser.parse_known_args(args)
    for path in parsed.paths:
        try:
            if fs.isdir(path):
                if not parsed.r:
                    raise TerminalError(f"rm: cannot remove '{path}': Is a directory")

                raise TerminalError("rm: recursive remove not fully implemented")
            else:
                fs.remove(path)
        except FileNotFoundError:
            raise TerminalError(
                f"rm: cannot remove '{path}': No such file or directory"
            )
        except TerminalError:
            raise
        except Exception as e:
            raise TerminalError(f"rm: {path}: {e}")
