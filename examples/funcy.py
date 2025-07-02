"""
Function Generation

Agent generates executable Python functions that can be called directly in your
program. Demonstrates runtime interoperability beyond JSON serialization.
"""

import math
from typing import Callable

from agex import Agent, Versioned

funcy_agent = Agent(primer="You are great at providing custom functions to the user.")
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

    # build a function to find next prime
    fn = fn_builder("a fn for the first prime larger than a given number.", state=state)

    # the function is callable in native python
    print(fn(500000))
    # 500009

    # agent remembers existing context and builds related function
    fn = fn_builder("Okay, now make it the next lower prime.", state=state)

    print(fn(500000))
    # 499979
