"""
One agent (the “optimizer”) refines a response, another (the “evaluator”)
critiques it until a response meets a quality criteria. As agents interop
easily with Python, we can manage the control flow in regular Python code.

For more details:
- https://langchain-ai.github.io/langgraph/tutorials/workflows/#evaluator-optimizer
- https://github.com/lastmile-ai/mcp-agent?tab=readme-ov-file#evaluator-optimizer
- https://www.anthropic.com/engineering/building-effective-agents
"""

from dataclasses import dataclass
from typing import Literal

from agex import Agent, connect_llm, pprint_tokens

client = connect_llm(provider="openai", model="gpt-5-nano", reasoning_effort="low")

optimizer = Agent(
    name="optimizer", primer="You create and hone jokes.", llm_client=client
)
evaluator = Agent(
    name="evaluator",
    primer="You critique jokes & suggest improvements.",
    llm_client=client,
)


@evaluator.cls
@optimizer.cls(constructable=False)
@dataclass
class Review:
    quality: Literal["good", "average", "bad"]
    feedback: str


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


def main():
    # create an initial joke
    joke = create_joke("pun about programming and fish", on_token=pprint_tokens)

    # hone the joke until it meets the quality criteria
    while (review := review_joke(joke, on_token=pprint_tokens)).quality != "good":
        joke = hone_joke(joke, review, on_token=pprint_tokens)

    print("Final joke:")
    print(joke)

    # Final joke:
    # Why do programmers prefer fishing? Because they love catching bugs and reeling in exceptions... and sometimes, they get caught in a loop!


if __name__ == "__main__":
    # Run with: python examples/evaluator_optimizer.py OR python -m examples.evaluator_optimizer
    main()
