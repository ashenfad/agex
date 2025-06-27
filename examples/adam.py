import logging
import math
from typing import Callable

from agex import Agent, Versioned

# It's helpful to see the agent's thinking process.
logging.basicConfig(level=logging.INFO)


# 1. Instantiate the agent.,
# We can optionally configure the LLM. If not provided, it uses smart
# defaults from environment variables or a config file.
# For this example, we'll let it use the defaults.
calculator_agent = Agent(
    primer="You are an expert at using a calculator. You are given a math problem and your goal is to return just the numeric answer.",
    max_iterations=5,
)


# 2. Give the agent capabilities.
# We'll register the entire `math` module, giving the agent access to
# functions like `sqrt`, `sin`, `cos`, constants like `pi`, etc.
# We'll give it the name "math" so the agent can `import math`.
calculator_agent.module(math, visibility="medium")


# 3. Define the task.
# The docstring is the prompt for the agent.
# The function signature tells the agent what inputs it will receive
# and what type of output it must produce.
@calculator_agent.task
def run_calculation(problem: str) -> float:  # type: ignore
    """
    Solve the user's math problem and return a number.
    """
    pass


@calculator_agent.task
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
