# Dogfood (Agents Creating Agents)

An “architect” agent uses the agex API to create a brand‑new specialist agent at runtime and returns its callable task.

## Setup

```python
import math
from typing import Callable
from agex import Agent

from dogfood_primer import PRIMER  # Primer coaching the architect pattern

architect = Agent(name="architect", primer=PRIMER)
architect.cls(Agent, visibility="medium")     # eat our own dogfood
architect.module(math, visibility="low")       # shareable capability
```

## Task: create a specialist and return a callable

```python
@architect.task
def create_specialist(prompt: str) -> Callable:  # type: ignore[return-value]
    """Create an agent task fn given a prompt."""
    pass
```

## Use it

```python
# Ask the architect to create a math solver specialist
math_solver = create_specialist("please create an agent that can solve math problems")
print(math_solver("4x + 5 = 13"))
# ... step-by-step solution ...
```

## Why it’s compelling
- Recursive capability building without leaving Python
- Specialists are just callables for the orchestrator
- With persistent state and future local registries, specialists can survive restarts

—

Source: [https://github.com/ashenfad/agex/blob/main/examples/dogfood.py](https://github.com/ashenfad/agex/blob/main/examples/dogfood.py)
Primer: [https://github.com/ashenfad/agex/blob/main/examples/dogfood_primer.py](https://github.com/ashenfad/agex/blob/main/examples/dogfood_primer.py)
