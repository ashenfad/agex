"""
State URI resolution for remote execution.

Handles translation of state URIs (e.g., "disk://session_id") to actual
Staged state objects.
"""

import os
from urllib.parse import urlparse

from kvgit import Staged, VersionedKV

from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Disk


class InvalidStateURIError(ValueError):
    """Raised when a state URI is invalid or cannot be resolved."""

    pass


def resolve_state_uri(
    uri: str,
    base_path: str = "/var/agex/state",
) -> Staged:
    """
    Resolve a state URI to a Staged state object.

    Currently supports:
    - disk://session_id: Disk-backed state in base_path/session_id

    Args:
        uri: The state URI (e.g., "disk://my_session")
        base_path: Base path for disk:// URIs (server-configured)

    Returns:
        A Staged state object

    Raises:
        InvalidStateURIError: If the URI is malformed or the scheme is unsupported
    """
    try:
        parsed = urlparse(uri)
    except Exception as e:
        raise InvalidStateURIError(f"Malformed URI: {uri}") from e

    scheme = parsed.scheme

    if scheme == "disk":
        return _resolve_disk_uri(parsed, base_path)

    raise InvalidStateURIError(f"Unsupported state scheme: {scheme}")


def _resolve_disk_uri(parsed, base_path: str) -> Staged:
    """
    Resolve a disk:// URI to a Staged state backed by disk storage.

    Security: Paths are normalized and sandboxed to the base_path directory.
    """
    # Extract session ID from netloc (disk://session_id)
    session_id = parsed.netloc

    if not session_id:
        raise InvalidStateURIError("disk:// URI must specify a session ID")

    # Normalize and sandbox the path
    # Prevent path traversal attacks
    safe_session_id = os.path.basename(session_id)  # Strip any path components
    if safe_session_id != session_id:
        raise InvalidStateURIError(
            f"Invalid session ID (path traversal attempt): {session_id}"
        )

    # Validate characters (alphanumeric, underscores, hyphens)
    if not all(c.isalnum() or c in "_-" for c in session_id):
        raise InvalidStateURIError(
            f"Session ID must be alphanumeric with underscores/hyphens: {session_id}"
        )

    # Construct full path
    full_path = os.path.join(base_path, session_id)

    # Double-check it's still under base_path (belt and suspenders)
    real_base = os.path.realpath(base_path)
    real_full = os.path.realpath(full_path)
    if not real_full.startswith(real_base):
        raise InvalidStateURIError(
            f"Path traversal prevented for session: {session_id}"
        )

    # Create the Staged state
    return Staged(
        VersionedKV(Disk(full_path)), encoder=_agex_encoder, decoder=_agex_decoder
    )
