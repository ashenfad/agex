"""
agex.server - Remote execution server for agex agents.

This package provides the server-side infrastructure for remote agent execution.
Requires optional dependencies: fastapi, uvicorn, sse-starlette, cloudpickle.

Install with: pip install agex[server]
"""

from .app import create_app, run_server
from .state import InvalidStateURIError, resolve_state_uri

__all__ = [
    "create_app",
    "run_server",
    "resolve_state_uri",
    "InvalidStateURIError",
]
