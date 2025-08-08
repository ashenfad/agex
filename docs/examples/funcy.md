# Funcy

Generate and return real Python callables from natural language prompts. No schemas; just use the function.

## Setup

```python
import math
from typing import Callable
from agex import Agent, Versioned

funcy_agent = Agent(primer="You are great at providing custom functions to the user.")
funcy_agent.module(math, visibility="low")
```

## Define the task (agent implements it)

```python
@funcy_agent.task
def fn_builder(prompt: str) -> Callable:  # type: ignore[return-value]
    """Build a callable function from a text prompt."""
    pass
```

## Build and use functions

```python
state = Versioned()  # keep context across calls

# Build a function that returns the first prime greater than n
next_prime = fn_builder("a fn for the first prime larger than a given number.", state=state)
print(next_prime(500000))
# 500009

# Build a related function leveraging context
prev_prime = fn_builder("Okay, now make it the next lower prime.", state=state)
print(prev_prime(500000))
# 499979
```

## Why this is different
- Returns an actual Python callable you can pass anywhere (sort keys, compose APIs).
- Works with your existing libraries — the agent composes them in code.
- Persistent context allows progressive capability building.

—

Source: https://github.com/ashenfad/agex/blob/main/examples/funcy.py
