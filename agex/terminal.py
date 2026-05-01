"""Types and helpers for agex terminal command registration.

Hosts register custom shell commands for ``terminal_action`` via two
sibling APIs on :class:`~agex.agent.Agent`:

- :meth:`~agex.agent.Agent.terminal_command` — decorator-style for the
  common case where the handler only needs the termish-shape context
  (args, stdin, stdout, fs).
- :meth:`~agex.agent.Agent.terminal_command_factory` — for handlers
  that need agex per-action runtime context (state, vfs).  The factory
  is called once per ``terminal_action`` with a fresh
  :class:`TerminalRuntime` and returns a termish-shape handler.

The split keeps :class:`TerminalContext` minimal and trivially
testable for the 95% case while giving complex handlers (like
``register_git``) a typed escape hatch for the agex internals they
need.

Termish's :class:`~termish.context.CommandContext`,
:class:`~termish.context.CommandResult`, and
:class:`~termish.errors.CommandFunc` are re-exported here so handler
authors can import their full toolset from a single place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TextIO

# Re-exports — handler authors should be able to import everything
# they need from agex.terminal without reaching into termish.
from termish.context import CommandContext, CommandResult
from termish.errors import CommandFunc

if TYPE_CHECKING:
    from termish.fs import FileSystem


__all__ = [
    "TerminalContext",
    "TerminalRuntime",
    "CommandContext",
    "CommandResult",
    "CommandFunc",
    "RESERVED_TERMINAL_NAMES",
]


# Names agex owns structurally and refuses to let user registrations
# shadow.  Currently just the ``python`` command — agex's bridge into
# nested ``python_action`` execution from a terminal pipeline.
# Termish builtins (ls, cat, grep, ...) are NOT in this set; termish
# explicitly supports user-injected commands overriding builtins, and
# agex inherits that contract.
RESERVED_TERMINAL_NAMES: frozenset[str] = frozenset({"python"})


@dataclass
class TerminalContext:
    """Per-invocation context for a terminal command handler.

    Mirrors the relevant subset of termish's
    :class:`~termish.context.CommandContext`.  Handlers needing agex
    per-action runtime (state, vfs) should register via
    :meth:`~agex.agent.Agent.terminal_command_factory` instead, which
    additionally provides a :class:`TerminalRuntime`.
    """

    args: list[str]
    """Parsed arguments (NOT including the command name)."""

    stdin: TextIO
    """Piped input from the previous pipeline stage, or empty."""

    stdout: TextIO
    """Write output here.  The pipeline captures this and forwards
    it to the next stage or final output."""

    fs: "FileSystem"
    """The agent's filesystem at the time of this action."""

    def fail(self, message: str, exit_code: int = 1) -> CommandResult:
        """Convenience: build a non-zero-exit :class:`CommandResult`."""
        return CommandResult(exit_code=exit_code, stderr=message)


@dataclass
class TerminalRuntime:
    """Per-action runtime context, exposed only to factory-registered
    terminal commands.  agex constructs a fresh instance for every
    ``terminal_action`` invocation; factories close over it to thread
    runtime values into their returned handler.
    """

    fs: "FileSystem"
    """The agent's filesystem at the time of this action."""

    state: Any | None = None
    """Per-action :class:`~agex.state.Staged` (or equivalent) for
    handlers that need to commit, peek at staged writes, etc.  ``None``
    when the action has no associated state (rare; mostly tests)."""

    vfs: Any | None = None
    """The :class:`~monkeyfs.VirtualFS` instance, when distinct from
    ``fs``.  Handlers that need internal VFS APIs (path encoding,
    metadata) use this; most handlers should just use ``fs``."""


# Visibility shares the existing high/medium/low vocabulary used by
# ``agent.fn`` / ``agent.cls`` / ``agent.module``.
TerminalVisibility = Literal["high", "medium", "low"]


@dataclass
class TerminalCommandRegistration:
    """Internal record for a registered terminal command.

    Stored on ``Agent._terminal_commands``; agex's
    ``build_terminal_commands`` consumes these to construct termish
    ``CommandFunc`` wrappers per action.

    Direct construction is not part of the public API — register
    commands via :meth:`~agex.agent.Agent.terminal_command` or
    :meth:`~agex.agent.Agent.terminal_command_factory` instead.
    """

    name: str
    handler: Callable[..., Any]
    """For ``kind="simple"`` this is a ``(TerminalContext) -> CommandResult | None``.
    For ``kind="factory"`` this is a ``(TerminalRuntime) -> CommandFunc``."""

    kind: Literal["simple", "factory"]
    visibility: TerminalVisibility = "high"
    docstring: str | None = None
