import math
from typing import Callable

from agex import Agent, Versioned

funcy_agent = Agent(
    primer="You are great a building functions. You are given a task and your goal is to build a function that will solve the task.",
    max_iterations=5,
)


funcy_agent.module(math, visibility="low")


@funcy_agent.task
def fn_builder(prompt: str) -> Callable:  # type: ignore
    """
    Make a fn for the user. Return this fn with `exit_success(my_fn)`.
    """
    pass


# 4. Run the agent.
if __name__ == "__main__":
    state = Versioned()

    fn = fn_builder(
        "Make a fn that finds a prime larger than a given number.",
        state=state,  # type: ignore
    )

    print(fn(50000))
    print(fn(100000))
    print(fn(500000))

    fn = fn_builder("Okay, now make it the next lower prime.", state=state)  # type: ignore

    print(fn(50000))
    print(fn(100000))
    print(fn(500000))

    # The agent will try to solve this problem.
    # We expect it to understand the request, use the `math` module,
    # and return a floating-point number.
    # problem_to_solve = "What is the square root of 256, multiplied by pi?"
    # print(f"Solving problem: '{problem_to_solve}'")

    # result = run_calculation(problem_to_solve)

    # print(f"\n✅ Agent finished. Result: {result}")
    # print(f"Type of result: {type(result)}")
