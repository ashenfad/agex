"""Gated socket implementation for network access control.

Provides method-level patching of socket.socket that checks the network_allowed
context variable before performing network operations.
"""

from __future__ import annotations

import socket
from typing import Any

from .context import network_allowed


class SandboxError(PermissionError):
    """Raised when network access is denied by the sandbox."""

    pass


# Socket families that represent network access (vs local IPC)
_NETWORK_FAMILIES = {socket.AF_INET, socket.AF_INET6}


def _is_network_socket(sock: socket.socket) -> bool:
    """Check if a socket is a network socket (vs local/unix socket).

    We only block network sockets (AF_INET, AF_INET6). Local sockets like
    AF_UNIX are used by asyncio internally for self-pipe tricks and must
    be allowed to pass through.

    Args:
        sock: The socket to check.

    Returns:
        True if this is a network socket that should be gated.
    """
    try:
        return sock.family in _NETWORK_FAMILIES
    except Exception:
        # If we can't determine the family, be conservative and gate it
        return True


def _check_network_allowed(operation: str, sock: socket.socket | None = None) -> None:
    """Check if network access is currently allowed.

    Args:
        operation: Name of the operation being attempted (for error message).
        sock: Optional socket to check. If provided and not a network socket,
              the operation is allowed regardless of network_allowed context.

    Raises:
        SandboxError: If network access is not allowed.
    """
    # Allow local sockets (AF_UNIX) - used by asyncio for self-pipe
    if sock is not None and not _is_network_socket(sock):
        return

    if not network_allowed.get():
        raise SandboxError(
            f"Network access denied: {operation}() blocked by sandbox. "
            f"Register function with network_access=True to allow network operations."
        )


# Store original socket methods for patching
_originals: dict[str, Any] = {}


def _patched_connect(self: socket.socket, address: Any) -> None:
    """Patched socket.connect that checks network permission."""
    _check_network_allowed("connect", self)
    return _originals["connect"](self, address)


def _patched_connect_ex(self: socket.socket, address: Any) -> int:
    """Patched socket.connect_ex that checks network permission."""
    _check_network_allowed("connect_ex", self)
    return _originals["connect_ex"](self, address)


def _patched_bind(self: socket.socket, address: Any) -> None:
    """Patched socket.bind that checks network permission."""
    _check_network_allowed("bind", self)
    return _originals["bind"](self, address)


def _patched_listen(self: socket.socket, backlog: int = 0) -> None:
    """Patched socket.listen that checks network permission."""
    _check_network_allowed("listen", self)
    return _originals["listen"](self, backlog)


def _patched_accept(self: socket.socket) -> tuple[socket.socket, Any]:
    """Patched socket.accept that checks network permission."""
    _check_network_allowed("accept", self)
    return _originals["accept"](self)


def _patched_send(self: socket.socket, data: bytes, flags: int = 0) -> int:
    """Patched socket.send that checks network permission."""
    _check_network_allowed("send", self)
    return _originals["send"](self, data, flags)


def _patched_sendall(self: socket.socket, data: bytes, flags: int = 0) -> None:
    """Patched socket.sendall that checks network permission."""
    _check_network_allowed("sendall", self)
    return _originals["sendall"](self, data, flags)


def _patched_sendto(self: socket.socket, data: bytes, *args: Any) -> int:
    """Patched socket.sendto that checks network permission."""
    _check_network_allowed("sendto", self)
    return _originals["sendto"](self, data, *args)


def _patched_recv(self: socket.socket, bufsize: int, flags: int = 0) -> bytes:
    """Patched socket.recv that checks network permission."""
    _check_network_allowed("recv", self)
    return _originals["recv"](self, bufsize, flags)


def _patched_recvfrom(
    self: socket.socket, bufsize: int, flags: int = 0
) -> tuple[bytes, Any]:
    """Patched socket.recvfrom that checks network permission."""
    _check_network_allowed("recvfrom", self)
    return _originals["recvfrom"](self, bufsize, flags)


def _patched_recv_into(
    self: socket.socket, buffer: Any, nbytes: int = 0, flags: int = 0
) -> int:
    """Patched socket.recv_into that checks network permission."""
    _check_network_allowed("recv_into", self)
    return _originals["recv_into"](self, buffer, nbytes, flags)


def _patched_recvfrom_into(
    self: socket.socket, buffer: Any, nbytes: int = 0, flags: int = 0
) -> tuple[int, Any]:
    """Patched socket.recvfrom_into that checks network permission."""
    _check_network_allowed("recvfrom_into", self)
    return _originals["recvfrom_into"](self, buffer, nbytes, flags)


def _patched_sendfile(
    self: socket.socket, file: Any, offset: int = 0, count: int | None = None
) -> int:
    """Patched socket.sendfile that checks network permission."""
    _check_network_allowed("sendfile", self)
    return _originals["sendfile"](self, file, offset, count)


# Store original getaddrinfo
_original_getaddrinfo: Any = None


def _patched_getaddrinfo(
    host: Any,
    port: Any,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> list[tuple[Any, ...]]:
    """Patched socket.getaddrinfo that checks network permission.

    DNS resolution is gated because it's often the first network operation
    and can leak information even if connections are blocked.
    """
    _check_network_allowed("getaddrinfo")
    return _original_getaddrinfo(host, port, family, type, proto, flags)
