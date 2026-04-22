"""Validators for XML-format file and edit actions.

These run while the tokenizer / response builder is assembling actions
from the stream. They fail fast with descriptive messages so the agent
loop can recover by reporting the error and letting the agent retry.
"""

import os
from typing import Literal

from agex.llm.core import ResponseParseError

from .tags import VALID_FILE_MODES


def validate_file_path(path: str) -> str:
    """Validate a file path from a ``<FILE>`` tag.

    Returns the stripped path. Raises :class:`ResponseParseError` on
    empty input, null bytes, or path traversal (``..``).
    """
    if not path or not path.strip():
        raise ResponseParseError("Empty path in <FILE> tag")

    path = path.strip()

    # Reject null bytes (can cause issues in some contexts).
    if "\x00" in path:
        raise ResponseParseError(f"Invalid characters in <FILE> path: {path!r}")

    # Reject path traversal attempts for clearer error messages
    # (VFS would handle this, but failing early is clearer).
    normalized = os.path.normpath(path)
    if normalized.startswith("..") or "/.." in normalized:
        raise ResponseParseError(f"Path traversal not allowed in <FILE> tag: {path}")

    return path


def validate_file_mode(mode: str, path: str) -> Literal["write", "append"]:
    """Validate the ``mode`` attribute of a ``<FILE>`` tag.

    Returns ``"write"`` or ``"append"``. Raises :class:`ResponseParseError`
    for any other value. ``path`` is used for error context.
    """
    mode = mode.lower().strip()
    if mode not in VALID_FILE_MODES:
        raise ResponseParseError(
            f"Invalid mode '{mode}' for <FILE path=\"{path}\">. "
            f"Must be 'write' or 'append'."
        )
    return mode  # type: ignore[return-value]


def validate_edit_search(path: str, search: str) -> str:
    """Validate a ``<SEARCH>`` inside ``<EDIT>``. Returns the search
    string unmodified (whitespace is significant for matching).

    Raises :class:`ResponseParseError` if empty.
    """
    # Don't strip — whitespace may be significant.
    if not search:
        raise ResponseParseError(f'Empty <SEARCH> in <EDIT path="{path}">')
    return search
