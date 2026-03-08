"""
Chapter Demo — Agent-Directed Context Compaction

Demonstrates how agents manage their own context by closing completed
stretches of work into named chapters. The agent tells several short
stories, the framework triggers chaptering when context grows large,
and then we ask the agent to recall details from chaptered history.

Run with: python examples/chapters.py
"""

from agex import Agent, connect_llm, connect_state, pprint_events, pprint_tokens

agent = Agent(
    name="storyteller",
    primer=(
        "You are a creative storyteller. When asked to tell a story, "
        "tell a vivid, detailed short story (2-3 paragraphs) with "
        "specific character names, places, and events."
    ),
    # llm=connect_llm(provider="anthropic", model="claude-haiku-4-5"),
    llm=connect_llm(provider="gemini", model="gemini-3-flash-preview"),
    state=connect_state(type="versioned", storage="memory"),
    # Low water marks to trigger chaptering quickly
    log_high_water_tokens=4000,
    log_low_water_tokens=2000,
    max_iterations=10,
)


@agent.task
def tell_story(prompt: str) -> str:
    """Tell a short, vivid story based on the prompt. Return the story text."""
    pass


@agent.task
def recall(question: str) -> str:
    """
    Answer a question about stories you've told previously.
    If the details were chaptered, you can browse /chapters to find them.
    Return your answer as a string.
    """
    pass


def main():
    stories = [
        "Tell a story about a lighthouse keeper who discovers a message in a bottle.",
        "Tell a story about a baker who accidentally invents a cake that makes people sing.",
        "Tell a story about an astronaut who finds a garden growing on the moon.",
    ]

    collected = []
    for i, prompt in enumerate(stories, 1):
        print(f"\n{'=' * 60}")
        print(f"Story {i}: {prompt}")
        print("=" * 60)
        story = tell_story(prompt, on_event=pprint_events, on_token=pprint_tokens)
        collected.append(story)
        print(f"\n--- Story {i} result ({len(story)} chars) ---")
        print(story[:200] + "..." if len(story) > 200 else story)

    # Now ask the agent to recall details — some should be in chapters
    print(f"\n{'=' * 60}")
    print("RECALL: Asking about details from earlier stories...")
    print("=" * 60)
    answer = recall(
        "What was the name of the lighthouse keeper from the first story? "
        "And how many paragraphs were in the story?",
        on_event=pprint_events,
        on_token=pprint_tokens,
    )
    print("\n--- Recall answer ---")
    print(answer)


if __name__ == "__main__":
    main()
