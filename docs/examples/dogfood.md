# Dogfood (Agents Creating Agents)

An “architect” agent uses the agex API to create a brand‑new specialist agent at runtime and returns its callable task.

Create an agent and register the agex library itself:

```python
import math
from typing import Callable
from agex import Agent

from dogfood_primer import PRIMER  # Primer coaching the architect pattern

architect = Agent(name="architect", primer=PRIMER)
architect.cls(Agent, visibility="medium")     # eat our own dogfood!
architect.module(math, visibility="low")      # shareable capability
```

A task function for creating specialist agents (returns a sub-agent task fn):

```python
@architect.task
def create_specialist(prompt: str) -> Callable:  # type: ignore[return-value]
    """Create an agent task fn given a prompt."""
    pass
```

Ask the specialist to make a new agent:

```python
# Ask the architect to create a math solver specialist
math_solver = create_specialist("please create an agent that can solve math problems")
# ----------------------------------------------
# actual `create_specialist` agent code for the task:
# ----------------------------------------------
# with Agent() as math_solver_agent:
#     import math
#     # Register math module capabilities with the agent
#     math_solver_agent.module(math)
#
#     def solve_math_problem(equation: str) -> str:
#         '''Solve a mathematical equation step by step.'''
#         pass  # Empty body for agent task
#
#     task_fn = math_solver_agent.task(solve_math_problem)
#
#     task_success(task_fn)
```

Give our new specialist some work:

```python
print(math_solver("4x + 5 = 13"))
# Subtract 5 from both sides: 4x = 13 - 5
# Divide both sides by 4: x = 8 / 4
# Therefore, x = 2.0

# ----------------------------------------------
# actual `math_solver` agent code for the task:
# ----------------------------------------------
# # From the equation, I see:
# coefficient_x = 4
# constant_term = 5
# right_side = 13
#
# # Step 1: Subtract 5 from both sides
# step1 = f"Subtract {constant_term} from both sides: {coefficient_x}x = {right_side} - {constant_term}"
# value_after_subtraction = right_side - constant_term
#
# # Step 2: Divide both sides by 4
# step2 = f"Divide both sides by {coefficient_x}: x = {value_after_subtraction} / {coefficient_x}"
#
# # Final calculation
# x_value = value_after_subtraction / coefficient_x
#
# # Compose the solution steps
# solution_steps = f"{step1}\n{step2}\nTherefore, x = {x_value}"
#
# task_success(solution_steps)
```

## Why it’s compelling:

- Recursive capability building without leaving Python
- Specialists are just callables for the orchestrator
- With persistent state and future local registries, specialists can survive restarts

—

Source: [https://github.com/ashenfad/agex/blob/main/examples/dogfood.py](https://github.com/ashenfad/agex/blob/main/examples/dogfood.py)
Primer: [https://github.com/ashenfad/agex/blob/main/examples/dogfood_primer.py](https://github.com/ashenfad/agex/blob/main/examples/dogfood_primer.py)
