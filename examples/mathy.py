"""
Mathematical Computing

Agent performs calculations using Python's math module and works with numerical
data. Demonstrates basic agent-module integration for computational tasks.
"""

import math

from agex import Agent

mathy_agent = Agent(primer="You are an expert at solving math problems.")

# medium viz shows function sigs but not docs to save context
mathy_agent.module(math, visibility="medium")


@mathy_agent.task
def run_calculation(problem: str) -> float:  # type: ignore[return-value]
    """Solve the mathematical problem and return the numeric result."""
    pass


@mathy_agent.task
def transform(prompt: str, numbers: list[float]) -> list[float]:  # type: ignore[return-value]
    """Transform a list of numbers based on a prompt."""
    pass


def main():
    """
    We run a calculation and then transform a list of numbers.
    """

    result = run_calculation("What is the square root of 256, multiplied by pi?")
    print(result)
    # 50.26548245743669

    nums = list(range(360))
    result = transform("Transform these degrees into radians", nums)
    print(len(result))
    print(result[-3:])
    # 360
    # [6.230825429619756, 6.2482787221397, 6.265732014659642]


if __name__ == "__main__":
    # Run with: python examples/mathy.py OR python -m examples.mathy
    main()
