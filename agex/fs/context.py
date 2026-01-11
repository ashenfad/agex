"""Context variables for filesystem isolation.

Shared context variables used by patching.py and filesystem implementations
to coordinate filesystem routing and prevent recursion loops.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from agex.fs.isolated import IsolatedFS
    from agex.fs.virtual import VirtualFS

# Context variables holding the current filesystems
current_isolated_fs: contextvars.ContextVar[IsolatedFS | None] = contextvars.ContextVar(
    "agex_current_isolated_fs", default=None
)

current_vfs: contextvars.ContextVar[VirtualFS | None] = contextvars.ContextVar(
    "agex_current_vfs", default=None
)


@contextmanager
def suspend_fs_interception() -> Iterator[None]:
    """Temporarily disable filesystem interception in the current context.

    Use this when implementing internal filesystem operations (like inside
    IsolatedFS) that need to perform real I/O without triggering the
    patched functions recursively.
    """
    token_iso = current_isolated_fs.set(None)
    token_vfs = current_vfs.set(None)
    try:
        yield
    finally:
        current_isolated_fs.reset(token_iso)
        current_vfs.reset(token_vfs)
