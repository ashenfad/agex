"""
Host abstraction for agent task execution.

Provides a unified interface for executing agent tasks locally or remotely.
"""

from typing import Literal

from .base import Host
from .http import HTTP, RemoteExecutionError, RemoteTimeoutError
from .local import Local
from .serialize import deserialize_agent, serialize_agent

__all__ = [
    "Host",
    "Local",
    "HTTP",
    "RemoteExecutionError",
    "RemoteTimeoutError",
    "connect_host",
    "serialize_agent",
    "deserialize_agent",
]


def connect_host(
    provider: Literal["local", "http"] = "local",
    **kwargs,
) -> Host:
    """
    Create an execution host.

    Args:
        provider: Host provider ("local" or "http")
        **kwargs: Provider-specific arguments

    Provider-specific kwargs:
        local:
            (no required args)

        http:
            url: str - Server URL (required)
            timeout: float - Request timeout (default 300.0)
            retries: int - Connection retry attempts (default 0)

    Returns:
        A Host instance

    Examples:
        # Local execution (default)
        host = connect_host()
        host = connect_host(provider="local")

        # Remote HTTP execution
        host = connect_host(provider="http", url="https://compute.example.com/execute")
        host = connect_host(provider="http", url="http://localhost:8000/execute", timeout=60.0)
    """
    if provider == "local":
        return Local()

    if provider == "http":
        if "url" not in kwargs:
            raise ValueError("HTTP host requires 'url' parameter")
        return HTTP(**kwargs)

    raise ValueError(f"Unknown host provider: {provider}. Available: local, http")
