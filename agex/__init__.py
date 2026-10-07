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

from .agent import Agent, Outcome, Session

__all__ = ["Agent", "Outcome", "Session"]
