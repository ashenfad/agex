"""A task that asks: ``task.needs_input`` stops it to ask the caller.

    AGEX_MODEL=anthropic:claude-sonnet-5-5 python examples/asks.py

``run`` returns the task's ``Outcome`` rather than its value. One that
ends ``needs_input`` carries the question as ``message`` and keeps its
world, and ``resume`` carries on there with the answer.
"""

import os
from dataclasses import dataclass

from agex import Agent


@dataclass
class Booking:
    restaurant: str
    day: str
    party: int


def main(agent: Agent, answer: str = "Four of us.") -> None:
    @agent.task
    def book(request: str) -> Booking:
        """A table booking that fits the request: a restaurant (any
        plausible name), the day and the party size. If the request
        doesn't say how many people are coming, ask with
        ``task.needs_input`` first."""
        ...

    out = book.run("Somewhere quiet for Friday dinner, Italian if you can.")
    if out.status == "needs_input":
        print(f"asked: {out.message}")
        print(f"answer: {answer}")
        out = out.resume(answer)
    if out.status == "success":
        b = out.value
        print(f"booked {b.restaurant} on {b.day} for {b.party}")
    else:
        print(f"[{out.status}] {out.message}")


if __name__ == "__main__":
    main(Agent(os.environ.get("AGEX_MODEL", "anthropic:claude-sonnet-5-5")))
