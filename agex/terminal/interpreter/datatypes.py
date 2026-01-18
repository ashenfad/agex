"""
Data types for the terminal interpreter.
"""

from typing import Callable, TextIO

from agex.fs.base import FileSystem


class TerminalError(Exception):
    """Raised when a terminal command execution fails."""

    def __init__(self, message: str, partial_output: str = ""):
        self.message = message
        self.partial_output = partial_output
        super().__init__(message)


# Command function signature:
# func(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None
# Raises TerminalError on failure.
CommandFunc = Callable[[list[str], TextIO, TextIO, FileSystem], None]
