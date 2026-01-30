"""Network access control for agex sandbox.

Provides gated socket implementation that blocks network access by default,
allowing it only for functions registered with network_access=True.
"""

from .context import allow_network, deny_network, network_allowed
from .socket import SandboxError

__all__ = [
    "SandboxError",
    "allow_network",
    "deny_network",
    "network_allowed",
]
