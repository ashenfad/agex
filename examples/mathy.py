import math

from agex import Agent

mathy_agent = Agent(
    primer="You are an expert at using a calculator. You are given a math problem and your goal is to return just the numeric answer.",
    max_iterations=5,
)
mathy_agent.module(math, visibility="medium")


@mathy_agent.task
def run_calculation(problem: str) -> float:  # type: ignore
    """
    Solve the user's math problem and return a number.
    """
    pass


if __name__ == "__main__":
    result = run_calculation(
        "What is the square root of 256, multiplied by pi?",
    )
    print(result)
