"""
Function Generation

Agent generates executable Python functions that can be called directly in your
program. Demonstrates runtime interoperability beyond JSON serialization.

https://asciinema.org/a/ZqYiNf6AJoskeVkcjPQAalgba
"""

import math
from typing import Callable

from agex import Agent, Versioned, connect_llm, pprint_tokens

funcy_agent = Agent(
    name="funcy",
    primer="You are great at providing custom functions to the user.",
    llm=connect_llm(provider="gemini", model="gemini-3-flash-preview"),
)
funcy_agent.module(math, visibility="low")


@funcy_agent.task
def fn_builder(prompt: str) -> Callable:  # type: ignore[return-value]
    """
    Build a callable function from a text prompt.
    """
    pass


def main():
    # Use versioned state to maintain context between agent calls
    state = Versioned()

    # build a function to find next prime
    print("\nPROMPT:", "a fn for the first prime larger than a given number.")
    fn = fn_builder(
        "a fn for the first prime larger than a given number.",
        state=state,
        on_token=pprint_tokens,
    )

    # ----------------------------------------------
    # actual `fn_builder` agent code for the task:
    # ----------------------------------------------
    # def first_prime_larger_than(n):
    #     def is_prime(num):
    #         if num <= 1:
    #             return False
    #         if num <= 3:
    #             return True
    #         if num % 2 == 0 or num % 3 == 0:
    #             return False
    #         i = 5
    #         while i * i <= num:
    #             if num % i == 0 or num % (i + 2) == 0:
    #                 return False
    #             i += 6
    #         return True
    #
    #     candidate = n + 1
    #     while True:
    #         if is_prime(candidate):
    #             return candidate
    #         candidate += 1
    #
    # task_success(first_prime_larger_than)

    # the function is callable in native python
    print("fn(500000) =", fn(500000), "\n")
    # 500009

    # agent remembers existing conversation context and builds related function
    print("\nPROMPT:", "Okay, now make it the next lower prime.")
    fn = fn_builder(
        "Okay, now make it the next lower prime.", state=state, on_token=pprint_tokens
    )
    print("fn(500000) =", fn(500000), "\n")
    # 499979


if __name__ == "__main__":
    # Run with: python examples/funcy.py OR python -m examples.funcy
    main()
