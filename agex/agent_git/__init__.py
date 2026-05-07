"""Agent-view git layer.

This package implements the "virtual" git an agent sees in
``terminal_action`` — branches, an index, and a commit log that operate
purely on VFS file content, leaving the underlying kvgit substrate
(event log, REPL namespace, agent memory) untouched.

Submodules:

* :mod:`agex.agent_git.metadata` — schema + I/O for the agent-view
  state blob (current branch, branch refs, staged-file index).
* :mod:`agex.agent_git.refs` — virtual-ancestry walks and ref
  resolution (``HEAD``, ``HEAD~N``, branch names, hash prefixes).
* :mod:`agex.agent_git.core` — the :class:`VirtualGit` class plus its
  result types (``AgentCommit``, ``Status``).
* :mod:`agex.agent_git.cli` — termish adapter (``register_git``,
  ``make_git_handler``).
"""

from .core import (
    AgentCommit,
    AgentGitError,
    BranchExists,
    BranchNotFound,
    BranchNotMerged,
    NothingToCommit,
    PathSpecError,
    PendingChanges,
    Status,
    UnbornBranch,
    VirtualGit,
    is_binary,
)
from .metadata import DEFAULT_BRANCH, METADATA_KEY, Metadata
from .refs import (
    InvalidRef,
    all_agent_commits,
    all_ancestors,
    merge_base,
    resolve_ref,
    virtual_parents,
    walk_virtual_ancestry,
)
from .refs import (
    is_agent_commit as is_agent_commit_hash,
)

__all__ = [
    # core
    "VirtualGit",
    "AgentCommit",
    "Status",
    "is_binary",
    # core errors
    "AgentGitError",
    "BranchExists",
    "BranchNotFound",
    "BranchNotMerged",
    "NothingToCommit",
    "PathSpecError",
    "PendingChanges",
    "UnbornBranch",
    # metadata
    "Metadata",
    "METADATA_KEY",
    "DEFAULT_BRANCH",
    # refs
    "InvalidRef",
    "resolve_ref",
    "walk_virtual_ancestry",
    "all_ancestors",
    "merge_base",
    "virtual_parents",
    "all_agent_commits",
    "is_agent_commit_hash",
]
