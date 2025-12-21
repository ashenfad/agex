"""
Async Function Generation

Demonstrates async task execution.

Agent-generated code remains synchronous while the framework handles
async transparently.

Usage: python examples/funcy_async.py
"""

import asyncio
import math
from typing import Callable

from agex import Agent, Versioned, connect_llm, pprint_tokens

funcy_agent = Agent(
    name="funcy_async",
    primer="You are great at providing custom functions to the user.",
    llm_client=connect_llm(provider="gemini", model="gemini-3-flash-preview"),
)
funcy_agent.module(math, visibility="low")


@funcy_agent.task
async def fn_builder(prompt: str) -> Callable:  # type: ignore[return-value]
    """
    Build a callable function from a text prompt.
    """
    pass


async def main():
    # Use versioned state to maintain context between agent calls
    state = Versioned()

    # Build a function to find next prime
    print("\nPROMPT:", "a fn for the first prime larger than a given number.")

    fn = await fn_builder(
        "a fn for the first prime larger than a given number.",
        state=state,
        on_token=pprint_tokens,
    )

    # The function is callable in native python
    print("\nfn(500000) =", fn(500000))
    # 500009

    # Agent remembers existing conversation context and builds related function
    print("\n\nPROMPT:", "Okay, now make it the next lower prime.")
    fn = await fn_builder(
        "Okay, now make it the next lower prime.",
        state=state,
        on_token=pprint_tokens,
    )
    print("\nfn(500000) =", fn(500000))
    # 499979


if __name__ == "__main__":
    # Run with: python examples/funcy_async.py
    asyncio.run(main())
