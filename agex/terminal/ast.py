"""
Abstract Syntax Tree (AST) definitions for the terminal command language.

This module defines the structured object model used to represent parsed
shell commands. It relies on frozen dataclasses for immutability.
"""

from dataclasses import dataclass, field
from typing import Literal

# Types of I/O redirection:
# <  : Input from file
# >  : Output to file (overwrite)
# >> : Output to file (append)
RedirectType = Literal["<", ">", ">>"]


@dataclass(frozen=True)
class Node:
    """Base class for all terminal AST nodes."""

    pass


@dataclass(frozen=True)
class Redirect(Node):
    """
    Represents an I/O redirection attached to a command.

    Attributes:
        type: The type of redirection (read, write, append).
        target: The target filename.
    """

    type: RedirectType
    target: str


@dataclass(frozen=True)
class Command(Node):
    """
    Represents a single executable command invocation.

    Example: `grep -r "pattern" .`

    Attributes:
        name: The name of the executable (e.g., 'grep').
        args: List of string arguments (e.g., ['-r', 'pattern', '.']).
        redirects: List of redirections associated with this command.
    """

    name: str
    args: list[str] = field(default_factory=list)
    redirects: list[Redirect] = field(default_factory=list)


@dataclass(frozen=True)
class Pipeline(Node):
    """
    Represents a sequence of commands connected by pipes.

    Example: `ls -la | grep .py`

    Attributes:
        commands: List of Command nodes to be executed in sequence,
                  with stdout of one piped to stdin of the next.
    """

    commands: list[Command]


@dataclass(frozen=True)
class Script(Node):
    """
    Represents a full script containing multiple pipelines.

    Example:
        cd /tmp
        ls -la | grep .log

    Attributes:
        pipelines: List of Pipeline nodes to be executed sequentially.
    """

    pipelines: list[Pipeline]
