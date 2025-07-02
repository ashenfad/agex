"""
Dogfooding Example: Agents Creating Agents

This example demonstrates agex's ability to "eat its own dogfood" by having
an agent build a brand-new agent on demand.
"""

import math
from typing import Callable

from dogfood_primer import PRIMER

from agex import Agent


def main():
    # Create an architect agent that can create other agents
    architect = Agent(name="architect", primer=PRIMER)

    # Register the Agent class so the architect can use it
    architect.cls(Agent)

    # Register math module so architect can pass it to child agents
    architect.module(math)

    @architect.task
    def create_specialist(prompt: str) -> Callable:  # type: ignore[return-value]
        """Create an agent task fn given a prompt."""
        pass

    #
    math_solver = create_specialist(
        "please create an agent that can solve math problems"
    )

    result = math_solver("4x + 5 = 13")
    print(result)
    # Subtract 5 from both sides: 4x = 13 - 5
    # Divide both sides by 4: x = 8 / 4
    # Therefore, x = 2.0


if __name__ == "__main__":
    # Run with: python examples/dogfood.py OR python -m examples.dogfood
    main()
