"""Context variables for network access control.

Provides context variable to track whether network access is currently allowed,
and context managers to control access during agent code evaluation.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

# Default: network access ALLOWED (so asyncio, pytest, etc. work normally)
# During agent code evaluation, we use deny_network() to block access.
network_allowed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agex_network_allowed", default=True
)


@contextmanager
def deny_network() -> Iterator[None]:
    """Deny network access in the current context.

    Use this context manager when entering agent code evaluation
    to block network operations by default.
    """
    token = network_allowed.set(False)
    try:
        yield
    finally:
        network_allowed.reset(token)


@contextmanager
def allow_network() -> Iterator[None]:
    """Temporarily allow network access in the current context.

    Use this context manager when calling functions that have been
    granted network_access=True permission.
    """
    token = network_allowed.set(True)
    try:
        yield
    finally:
        network_allowed.reset(token)
