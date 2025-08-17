# Mathy

Solve numerical problems by composing the Python `math` module directly — no JSON tools, no wrappers. The agent “thinks in code,” calling library primitives within a single turn.

Setup an agent with the math module:

```python
import math
from agex import Agent

mathy_agent = Agent(primer="You are an expert at solving math problems.")

# Show function signatures to the agent; skip long docs to save tokens
mathy_agent.module(math, visibility="medium")
```

Define tasks via function sigs (agents implement them when called):

```python
@mathy_agent.task
def run_calculation(problem: str) -> float:  # type: ignore[return-value]
    """Solve the mathematical problem and return the numeric result."""
    pass

@mathy_agent.task
def transform(prompt: str, numbers: list[float]) -> list[float]:  # type: ignore[return-value]
    """Transform a list of numbers based on a prompt."""
    pass
```

Call the task fns:

```python
# Single-turn composition across math primitives
print(run_calculation("What is the square root of 256, multiplied by pi?"))
# 50.2654824574...

nums = list(range(360))
print(transform("Transform these degrees into radians", nums))
# [... 6.2308254296, 6.2482787221, 6.2657320146]
```

Why this works:

- Code-as-action: the agent writes Python that calls `math` directly.
- Visibility control: signatures are enough for well-known libs.
- Minimal ceremony: you define intent (signature + docstring); the agent provides implementation at runtime.

—

Source: [https://github.com/ashenfad/agex/blob/main/examples/mathy.py](https://github.com/ashenfad/agex/blob/main/examples/mathy.py)
