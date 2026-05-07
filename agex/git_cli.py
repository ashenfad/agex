"""Backwards-compatible re-export of the agent-view git CLI.

The implementation lives in :mod:`agex.agent_git.cli`; this module
keeps the historic ``from agex.git_cli import register_git`` /
``make_git_handler`` import paths working without forcing callers to
follow the package split.
"""

from .agent_git.cli import make_git_handler, register_git

__all__ = ["register_git", "make_git_handler"]
