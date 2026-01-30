"""Network sandbox patching functions.

Provides functions to install and uninstall the network sandbox by patching
socket methods to check the network_allowed context variable.
"""

from __future__ import annotations

import socket
import threading

from . import socket as gated_socket_module
from .socket import (
    _originals,
    _patched_accept,
    _patched_bind,
    _patched_connect,
    _patched_connect_ex,
    _patched_getaddrinfo,
    _patched_listen,
    _patched_recv,
    _patched_recv_into,
    _patched_recvfrom,
    _patched_recvfrom_into,
    _patched_send,
    _patched_sendall,
    _patched_sendfile,
    _patched_sendto,
)

# Lock for thread-safe patch installation
_patch_lock = threading.Lock()
_patch_installed = False


def install_network_sandbox() -> None:
    """Install the network sandbox by patching socket methods.

    This function is idempotent - calling it multiple times has no additional effect.

    The sandbox patches socket.socket methods (connect, send, recv, etc.) to check
    the network_allowed context variable before proceeding. When network_allowed
    is False (during agent code evaluation), operations raise SandboxError.

    Note:
        This is called lazily from evaluate_program. Most major HTTP libraries
        (urllib3, requests, httpx, aiohttp) use dynamic lookups and will work
        correctly even with late patching.
    """
    global _patch_installed

    with _patch_lock:
        if _patch_installed:
            return

        # Store original socket methods
        _originals["connect"] = socket.socket.connect
        _originals["connect_ex"] = socket.socket.connect_ex
        _originals["bind"] = socket.socket.bind
        _originals["listen"] = socket.socket.listen
        _originals["accept"] = socket.socket.accept
        _originals["send"] = socket.socket.send
        _originals["sendall"] = socket.socket.sendall
        _originals["sendto"] = socket.socket.sendto
        _originals["recv"] = socket.socket.recv
        _originals["recvfrom"] = socket.socket.recvfrom
        _originals["recv_into"] = socket.socket.recv_into
        _originals["recvfrom_into"] = socket.socket.recvfrom_into
        _originals["sendfile"] = socket.socket.sendfile

        # Store original getaddrinfo
        gated_socket_module._original_getaddrinfo = socket.getaddrinfo

        # Install patched methods on socket class
        socket.socket.connect = _patched_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = _patched_connect_ex  # type: ignore[method-assign]
        socket.socket.bind = _patched_bind  # type: ignore[method-assign]
        socket.socket.listen = _patched_listen  # type: ignore[method-assign]
        socket.socket.accept = _patched_accept  # type: ignore[method-assign]
        socket.socket.send = _patched_send  # type: ignore[method-assign]
        socket.socket.sendall = _patched_sendall  # type: ignore[method-assign]
        socket.socket.sendto = _patched_sendto  # type: ignore[method-assign]
        socket.socket.recv = _patched_recv  # type: ignore[method-assign]
        socket.socket.recvfrom = _patched_recvfrom  # type: ignore[method-assign]
        socket.socket.recv_into = _patched_recv_into  # type: ignore[method-assign]
        socket.socket.recvfrom_into = _patched_recvfrom_into  # type: ignore[method-assign]
        socket.socket.sendfile = _patched_sendfile  # type: ignore[method-assign]

        # Install patched getaddrinfo
        socket.getaddrinfo = _patched_getaddrinfo

        _patch_installed = True


def uninstall_network_sandbox() -> None:
    """Uninstall the network sandbox, restoring original socket behavior.

    This function is idempotent - calling it when not installed has no effect.
    """
    global _patch_installed

    with _patch_lock:
        if not _patch_installed:
            return

        # Restore original socket methods
        if "connect" in _originals:
            socket.socket.connect = _originals["connect"]  # type: ignore[method-assign]
        if "connect_ex" in _originals:
            socket.socket.connect_ex = _originals["connect_ex"]  # type: ignore[method-assign]
        if "bind" in _originals:
            socket.socket.bind = _originals["bind"]  # type: ignore[method-assign]
        if "listen" in _originals:
            socket.socket.listen = _originals["listen"]  # type: ignore[method-assign]
        if "accept" in _originals:
            socket.socket.accept = _originals["accept"]  # type: ignore[method-assign]
        if "send" in _originals:
            socket.socket.send = _originals["send"]  # type: ignore[method-assign]
        if "sendall" in _originals:
            socket.socket.sendall = _originals["sendall"]  # type: ignore[method-assign]
        if "sendto" in _originals:
            socket.socket.sendto = _originals["sendto"]  # type: ignore[method-assign]
        if "recv" in _originals:
            socket.socket.recv = _originals["recv"]  # type: ignore[method-assign]
        if "recvfrom" in _originals:
            socket.socket.recvfrom = _originals["recvfrom"]  # type: ignore[method-assign]
        if "recv_into" in _originals:
            socket.socket.recv_into = _originals["recv_into"]  # type: ignore[method-assign]
        if "recvfrom_into" in _originals:
            socket.socket.recvfrom_into = _originals["recvfrom_into"]  # type: ignore[method-assign]
        if "sendfile" in _originals:
            socket.socket.sendfile = _originals["sendfile"]  # type: ignore[method-assign]

        # Restore original getaddrinfo
        if gated_socket_module._original_getaddrinfo is not None:
            socket.getaddrinfo = gated_socket_module._original_getaddrinfo

        # Clear stored originals
        _originals.clear()
        gated_socket_module._original_getaddrinfo = None

        _patch_installed = False


def is_network_sandbox_installed() -> bool:
    """Check if the network sandbox is currently installed.

    Returns:
        True if the sandbox is installed, False otherwise.
    """
    return _patch_installed
