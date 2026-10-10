"""agex: agents that work in a versioned world.

agex is a loop over nontainer. A session drives a workspace turn by
turn, and a task runs on a fork of one and hands back a typed value;
both keep their conversation in the workspace's branch, so a checkout
rewinds memory with the files and a fork carries it.

    from agex import Agent

    agent = Agent("anthropic:claude-sonnet-5-5", primer="You build study apps.")
    chat = agent.session(ws)
    outcome = chat.say("make a flashcard app")

This is the 0.13 rebuild, in progress. The 0.12 line is at the
``v0.12.4`` tag.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import Agent, Outcome, Session
    from .agent_tasks import agent_tasks
    from .stand_in import stand_in_tasks
    from .task import (
        NeedsInput,
        Task,
        TaskError,
        TaskFailed,
        TaskInterrupted,
        TaskSpec,
    )

_HOMES = {
    "Agent": ".agent",
    "Outcome": ".agent",
    "Session": ".agent",
    "agent_tasks": ".agent_tasks",
    "stand_in_tasks": ".stand_in",
    "NeedsInput": ".task",
    "Task": ".task",
    "TaskError": ".task",
    "TaskFailed": ".task",
    "TaskInterrupted": ".task",
    "TaskSpec": ".task",
}
"""Where each name lives, imported on first use: code that runs agent
work in a worker or a dud guest imports ``agex.stubs``, and importing
the package for it must not bring in the loop and its providers."""


def __getattr__(name: str) -> Any:
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module 'agex' has no attribute {name!r}")
    value = getattr(importlib.import_module(home, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Agent",
    "Outcome",
    "Session",
    "agent_tasks",
    "stand_in_tasks",
    "NeedsInput",
    "Task",
    "TaskError",
    "TaskFailed",
    "TaskInterrupted",
    "TaskSpec",
]
