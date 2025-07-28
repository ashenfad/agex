"""
Dogfooding Example: Agents Creating Agents

This example demonstrates agex's ability to "eat its own dogfood" by having
an agent build a brand-new agent on demand.

Note: This example works within a single process. Persistence of dynamically
created agents across process boundaries is a planned roadmap item.

Note: This example was tested with `gpt-4.1-nano`, highlighting how `agex`'s
"micro-DSL" approach—providing a focused set of capabilities—can guide even
smaller models to success on complex tasks.
"""

import math
from typing import Callable

from dogfood_primer import PRIMER

from agex import Agent

# create an architect agent that can create other agents
architect = Agent(name="architect", primer=PRIMER)

# register the Agent class so the architect can use it...
# this is where we eat our own dogfood!
architect.cls(Agent, visibility="medium")

# register math module so the architect may share it with child agents
architect.module(math, visibility="low")


@architect.task
def create_specialist(prompt: str) -> Callable:  # type: ignore[return-value]
    """Create an agent task fn given a prompt."""
    pass


def main():
    # ask the architect to create a brand-new specialist agent
    math_solver = create_specialist(
        "please create an agent that can solve math problems"
    )

    # ----------------------------------------------
    # actual `create_specialist` agent code for the task:
    # ----------------------------------------------
    # with Agent() as math_solver_agent:
    #     import math
    #     # Register math module capabilities with the agent
    #     math_solver_agent.module(math)
    #
    #     def solve_math_problem(equation: str) -> str:
    #         '''Solve a mathematical equation step by step.'''
    #         pass  # Empty body for agent task
    #
    #     task_fn = math_solver_agent.task(solve_math_problem)
    #
    #     task_success(task_fn)

    result = math_solver("4x + 5 = 13")
    print(result)
    # Subtract 5 from both sides: 4x = 13 - 5
    # Divide both sides by 4: x = 8 / 4
    # Therefore, x = 2.0

    # ----------------------------------------------
    # actual `math_solver` agent code for the task:
    # ----------------------------------------------
    # # From the equation, I see:
    # coefficient_x = 4
    # constant_term = 5
    # right_side = 13
    #
    # # Step 1: Subtract 5 from both sides
    # step1 = f"Subtract {constant_term} from both sides: {coefficient_x}x = {right_side} - {constant_term}"
    # value_after_subtraction = right_side - constant_term
    #
    # # Step 2: Divide both sides by 4
    # step2 = f"Divide both sides by {coefficient_x}: x = {value_after_subtraction} / {coefficient_x}"
    #
    # # Final calculation
    # x_value = value_after_subtraction / coefficient_x
    #
    # # Compose the solution steps
    # solution_steps = f"{step1}\n{step2}\nTherefore, x = {x_value}"
    #
    # task_success(solution_steps)


if __name__ == "__main__":
    # Run with: python examples/dogfood.py OR python -m examples.dogfood
    main()
