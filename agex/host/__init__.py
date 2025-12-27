"""
Host abstraction for agent task execution.

Provides a unified interface for executing agent tasks locally or remotely.
"""

from typing import Literal

from .base import Host
from .http import HTTP, RemoteExecutionError, RemoteTimeoutError
from .local import Local
from .runner import (
    aexecute_task,
    arun_remote_task,
    execute_task,
    prepare_agent,
    run_remote_task,
)
from .serialize import deserialize_agent, serialize_agent

__all__ = [
    # Host abstraction
    "Host",
    "Local",
    "HTTP",
    "RemoteExecutionError",
    "RemoteTimeoutError",
    "connect_host",
    # Serialization
    "serialize_agent",
    "deserialize_agent",
    # Remote execution (for server-side use)
    "prepare_agent",
    "execute_task",
    "aexecute_task",
    "run_remote_task",
    "arun_remote_task",
]


def connect_host(
    provider: Literal["local", "http", "modal"] = "local",
    **kwargs,
) -> Host:
    """
    Create an execution host.

    Args:
        provider: Host provider ("local", "http", or "modal")
        **kwargs: Provider-specific arguments

    Provider-specific kwargs:
        local:
            (no required args)

        http:
            url: str - Server URL (required)
            timeout: float - Request timeout (default 300.0)
            retries: int - Connection retry attempts (default 0)

        modal:
            app: str - Modal app name (optional, defaults to "agex-{agent_name}-{fingerprint}")
            volume: str - Modal volume name for state storage
            secrets: str | list[str] - Modal secret names (required, e.g. "llm-keys")
            gpu: str - GPU type (e.g., "A10G", "T4", "A100")
            memory: int - Memory in MB
            timeout: float - Execution timeout (default 300.0)
            detach: bool - Verify deploy/detach mode (default True)
            scaledown_window: int - Keep containers warm for N seconds (default 300)

    Returns:
        A Host instance

    Examples:
        # Local execution (default)
        host = connect_host()
        host = connect_host(provider="local")

        # Remote HTTP execution
        host = connect_host(provider="http", url="https://compute.example.com/execute")
        host = connect_host(provider="http", url="http://localhost:8000/execute", timeout=60.0)

        # Modal serverless execution
        host = connect_host(
            provider="modal",
            app="my-app",
            volume="agex-state",
            secrets=["llm-keys"],
            gpu="A10G",
        )
    """
    if provider == "local":
        return Local()

    if provider == "http":
        if "url" not in kwargs:
            raise ValueError("HTTP host requires 'url' parameter")
        return HTTP(**kwargs)

    if provider == "modal":
        try:
            from agex.host.modal import Modal
        except ModuleNotFoundError as e:
            if "modal" in str(e):
                raise ModuleNotFoundError(
                    "Modal host requires the 'modal' package. "
                    "Install it with: pip install agex[modal]"
                ) from None
            raise

        return Modal(**kwargs)

    raise ValueError(
        f"Unknown host provider: {provider}. Available: local, http, modal"
    )
