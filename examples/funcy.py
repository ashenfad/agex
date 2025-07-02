"""
Function Generation

Agent generates executable Python functions that can be called directly in your
program. Demonstrates runtime interoperability beyond JSON serialization.
"""

import math
from typing import Callable

from agex import Agent, Versioned

funcy_agent = Agent(
    primer="You are great a building functions. You are given a task and your goal is to build a function that will solve the task.",
    max_iterations=5,
)

funcy_agent.module(math, visibility="low")


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
        "Make a fn that finds a prime larger than a given number.", state=state
    )

    # the function is callable in native python
    print(fn(500000))
    # 500009

    # Second request: agent remembers context and builds related function
    fn = fn_builder("Okay, now make it the next lower prime.", state=state)

    print(fn(500000))
    # 499979
