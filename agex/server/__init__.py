"""
agex.server - Remote execution server for agex agents.

This package provides the server-side infrastructure for remote agent execution.
Requires optional dependencies: fastapi, uvicorn, sse-starlette, cloudpickle.

Install with: pip install agex[server]
"""

from .app import create_app, run_server
from .state import InvalidStateURIError, resolve_state_uri

# Create default app instance for convenience
# Can be run with: uvicorn agex.server:app
app = create_app()

__all__ = [
    "app",
    "create_app",
    "run_server",
    "resolve_state_uri",
    "InvalidStateURIError",
]
