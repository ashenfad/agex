"""
Parser for the terminal command language.

Converts shell-like command strings into the AST defined in agex.terminal.ast.
Uses shlex for lexical analysis (tokenization).
"""

import shlex
from typing import List, Optional

from .ast import Command, Pipeline, Redirect, Script


class ParseError(Exception):
    """Raised when the parser encounters invalid syntax."""

    pass


def to_script(text: str) -> Script:
    """
    Parse a command string into a Script AST node.

    Args:
        text: The shell command string.

    Returns:
        A Script node containing the parsed pipelines.

    Raises:
        ParseError: If the syntax is invalid.
    """
    if not text or not text.strip():
        return Script(pipelines=[])

    # Configure shlex to handle shell punctuation as separate tokens
    # punctuation_chars=True ensures "ls|grep" becomes ["ls", "|", "grep"]
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)

    # Treat newlines as tokens, not whitespace, so we can use them as separators
    lexer.whitespace = " \t\r"

    try:
        tokens = list(lexer)
    except ValueError as e:
        raise ParseError(f"Tokenization error: {e}") from e

    return _parse_tokens(tokens)


def _parse_tokens(tokens: List[str]) -> Script:
    """
    Convert a list of tokens into a Script.

    Structure:
    Script = Pipeline { (";" | NEWLINE) Pipeline }*
    Pipeline = Command { "|" Command }*
    Command = Word { Arg | Redirect }*
    """
    pipelines: List[Pipeline] = []
    current_pipeline_cmds: List[Command] = []

    # Iterator for consumption
    it = iter(tokens)

    # Current command build state
    cmd_name: Optional[str] = None
    cmd_args: List[str] = []
    cmd_redirects: List[Redirect] = []

    def flush_command():
        nonlocal cmd_name, cmd_args, cmd_redirects
        if cmd_name:
            current_pipeline_cmds.append(
                Command(name=cmd_name, args=cmd_args, redirects=cmd_redirects)
            )
        cmd_name = None
        cmd_args = []
        cmd_redirects = []

    def flush_pipeline():
        nonlocal current_pipeline_cmds
        flush_command()
        if current_pipeline_cmds:
            pipelines.append(Pipeline(commands=current_pipeline_cmds))
        current_pipeline_cmds = []

    try:
        while True:
            token = next(it)

            if token == ";" or token == "\n":
                flush_pipeline()
                continue

            elif token == "|":
                flush_command()
                if not current_pipeline_cmds and not cmd_name:
                    raise ParseError("Unexpected pipe '|' before command")
                continue

            elif token in (">", ">>", "<"):
                # Handle Redirect
                try:
                    target = next(it)
                    # Check if target is another operator
                    # Note: we check for \n here too
                    if target in (";", "|", ">", ">>", "<", "\n"):
                        raise ParseError(
                            f"Expected filename after '{token}', got '{target}'"
                        )
                except StopIteration:
                    raise ParseError(f"Expected filename after '{token}'")

                cmd_redirects.append(Redirect(type=token, target=target))  # type: ignore[arg-type]
                continue

            else:
                # Regular word (Command Name or Argument)
                if cmd_name is None:
                    cmd_name = token
                else:
                    cmd_args.append(token)

    except StopIteration:
        pass

    # Final flush
    flush_pipeline()

    return Script(pipelines=pipelines)
