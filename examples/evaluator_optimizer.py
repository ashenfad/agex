"""
One agent (the “optimizer”) refines a response, another (the “evaluator”)
critiques it until a response meets a quality criteria. As agents interop
easily with Python, we can manage the control flow in regular Python code.

For more details:
- https://langchain-ai.github.io/langgraph/tutorials/workflows/#evaluator-optimizer
- https://github.com/lastmile-ai/mcp-agent?tab=readme-ov-file#evaluator-optimizer
- https://www.anthropic.com/engineering/building-effective-agents

Note: This example was tested with gpt-4.1-nano to demonstrate that
even smaller LLMs can effectively use the agex framework.
"""

from dataclasses import dataclass
from typing import Literal

from agex import Agent, Versioned

optimizer = Agent(name="optimizer", primer="You create and hone jokes.")
evaluator = Agent(name="evaluator", primer="You critique jokes & suggest improvements.")


@evaluator.cls
@optimizer.cls
@dataclass
class Review:
    quality: Literal["good", "average", "bad"]
    feedback: str
    joke: str


@optimizer.task
def create_joke(topic: str) -> str:  # type: ignore[return-value]
    """Create a joke given a topic"""
    pass


@optimizer.task
def hone_joke(review: Review) -> str:  # type: ignore[return-value]
    """Hone a joke given feedback"""
    pass


@evaluator.task
def review_joke(joke: str) -> Review:  # type: ignore[return-value]
    """Judge a joke and suggest improvements by returning a Review"""
    pass


def main():
    state = Versioned()

    # create an initial joke
    joke = create_joke("pun about programming and fish", state=state)

    # hone the joke until it meets the quality criteria
    while (review := review_joke(joke, state=state)).quality != "good":
        joke = hone_joke(review, state=state)

    print("Final joke:")
    print(review.joke)

    print("Final feedback:")
    print(review.feedback)

    # Final joke:
    # Why do programmers prefer fishing? Because they love catching bugs and reeling in exceptions... and sometimes, they get caught in a loop!
    # Final feedback:
    # The joke creatively combines programming metaphors with fishing, making it relatable and humorous for programmers. To improve, consider sharpening the punchline for greater impact or adding a vivid image to make it more memorable.


if __name__ == "__main__":
    # Run with: python examples/evaluator_optimizer.py OR python -m examples.evaluator_optimizer
    main()
