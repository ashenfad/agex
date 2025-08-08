# Evaluator–Optimizer (Peer Collaboration)

Two agents collaborate as peers: one creates, the other critiques and suggests improvements, iterating until quality criteria are met.

## Setup

```python
from dataclasses import dataclass
from typing import Literal
from agex import Agent

optimizer = Agent(name="optimizer", primer="You create and hone jokes.")
evaluator = Agent(name="evaluator", primer="You critique jokes & suggest improvements.")

@evaluator.cls
@optimizer.cls(constructable=False)
@dataclass
class Review:
    quality: Literal["good", "average", "bad"]
    feedback: str
```

## Tasks (agent-implemented)

```python
@optimizer.task
def create_joke(topic: str) -> str:  # type: ignore[return-value]
    """Create a joke given a topic"""
    pass

@optimizer.task
def hone_joke(joke: str, review: Review) -> str:  # type: ignore[return-value]
    """Hone a joke given feedback"""
    pass

@evaluator.task
def review_joke(joke: str) -> Review:  # type: ignore[return-value]
    """Judge a joke and suggest improvements by returning a Review"""
    pass
```

## Loop until good

```python
joke = create_joke("pun about programming and fish")
while (review := review_joke(joke)).quality != "good":
    joke = hone_joke(joke, review)

print(joke)
```

## Why it’s useful
- Natural control flow in host Python
- Typed feedback loop via a shared `Review` dataclass
- Works well for refinement, critique, and QA

—

Source: https://github.com/ashenfad/agex/blob/main/examples/evaluator_optimizer.py
