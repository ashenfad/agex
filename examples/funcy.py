"""
Function Generation

Agents generate executable Python functions that can be called directly in your
program. Demonstrates runtime interoperability beyond JSON serialization.
"""

import math
from typing import Callable

from agex import Agent, Versioned

# 1. Create an agent specialized in building functions
funcy_agent = Agent(
    primer="You are great a building functions. You are given a task and your goal is to build a function that will solve the task.",
    max_iterations=5,
)

# 2. Register the math module with low visibility
# (available but not shown in agent view to save context)
funcy_agent.module(math, visibility="low")


# 3. Define a task that builds and returns functions
@funcy_agent.task
def fn_builder(prompt: str) -> Callable:  # type: ignore[return-value]
    """
    Build a callable function from a text prompt.
    """
    pass


def example():
    # Use versioned state to maintain context between agent calls
    state = Versioned()

    # First request: build a function to find next prime
    fn = fn_builder(
        "Make a fn that finds a prime larger than a given number.",
        state=state,  # type: ignore
    )

    # the function is callable in native python
    print(fn(50000))
    print(fn(100000))
    print(fn(500000))

    # Second request: agent remembers context and builds related function
    fn = fn_builder("Okay, now make it the next lower prime.", state=state)  # type: ignore[call-arg]

    print(fn(50000))
    print(fn(100000))
    print(fn(500000))
