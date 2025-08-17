# Funcy

Generate and return real Python callables from natural language prompts. No schemas; just use the function.

Setup the agent and let it use the math module:

```python
import math
from typing import Callable
from agex import Agent, Versioned

funcy_agent = Agent(primer="You are great at providing custom functions to the user.")
funcy_agent.module(math, visibility="low")
```

Define the task with a function sig, input/output types define the contract:

```python
@funcy_agent.task
def fn_builder(prompt: str) -> Callable:  # type: ignore[return-value]
    """Build a callable function from a text prompt."""
    pass
```

Ask for a prime-finder using state so the agent remembers across tasks:

```python
state = Versioned()  # keep context across calls

# Build a function that returns the first prime greater than n
next_prime = fn_builder(
    "a fn for the first prime larger than a given number.", state=state
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
```

Try out the prime finder and then ask for a variation:

```python
print(next_prime(500000))
# 500009

# Build a related function leveraging context
prev_prime = fn_builder("Okay, now make it the next lower prime.", state=state)
print(prev_prime(500000))
# 499979
```

Why this is different:

- Returns an actual Python callable you can pass anywhere (sort keys, compose APIs).
- Works with your existing libraries — the agent composes them in code.
- Persistent context allows progressive capability building.

—

Source: [https://github.com/ashenfad/agex/blob/main/examples/funcy.py](https://github.com/ashenfad/agex/blob/main/examples/funcy.py)
