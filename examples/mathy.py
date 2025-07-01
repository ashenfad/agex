"""
Mathematical Computing

Agents perform calculations using Python's math module and work with numerical
data. Demonstrates basic agent-module integration for computational tasks.
"""

import math

from agex import Agent

# 1. Create an agent with a clear role and iteration limit
mathy_agent = Agent(
    primer="You are an expert at using a calculator. You are given a math problem and your goal is to return just the numeric answer.",
    max_iterations=5,
)

# 2. Register the math module with medium visibility
# (shows function signatures but not docstrings to save context)
mathy_agent.module(math, visibility="medium")


# 3. Define agent tasks
@mathy_agent.task
def run_calculation(problem: str) -> float:  # type: ignore[return-value]
    """Solve the mathematical problem and return the numeric result."""
    pass


@mathy_agent.task
def transform_or_aggregate(prompt: str, numbers: list[float]) -> list[float] | float:  # type: ignore[return-value]
    """Transform or aggregate a list of numbers based on a prompt."""
    pass


def example():
    """
    We run a calculation and then transform a list of numbers.
    """

    result = run_calculation("What is the square root of 256, multiplied by pi?")
    print(result)

    nums = list(range(360))
    result = transform_or_aggregate("Transform these degrees into radians", nums)
    print(result[-3:])
