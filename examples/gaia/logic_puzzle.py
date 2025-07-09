"""
GAIA-style Logic Puzzle

Problem: You have 8 coins that look identical, but one is counterfeit and weighs
slightly less than the others. You have a balance scale that can compare weights.
What is the minimum number of weighings needed to guarantee finding the counterfeit
coin? Provide a strategy that achieves this minimum.

This demonstrates:
- Pure logical reasoning
- Mathematical optimization
- Strategic thinking
- Algorithm design
"""

import math
from typing import Literal

from agex import Agent

# Create agent focused on logical reasoning
agent = Agent(
    name="logic_solver",
    primer="You excel at logical reasoning and mathematical puzzles.",
)


agent.module(math, visibility="low")


@agent.task
def solve_coin_puzzle(num_coins: int, difference: Literal["heavier", "lighter"]) -> int:  # type: ignore[return-value]
    """
    Solve the classic coin puzzle: find the minimum number of weighings needed
    to identify the different coin among num_coins, where the different coin is
    either heavier or lighter than the others.
    """
    pass


@agent.task
def solve_card_arrangement(deck_size: int, question: str) -> int:  # type: ignore[return-value]
    """Solve card arrangement puzzles."""
    pass


@agent.task
def solve_logic_grid(clues: list[str], question: str) -> str:  # type: ignore[return-value]
    """
    Solve logic grid puzzles given a set of clues.
    """
    pass


def main():
    print("=== Coin Puzzle ===")
    result = solve_coin_puzzle(8, "lighter")
    print(f"Minimum weighings: {result} - (expected 3)")

    print("\n=== Josephus Problem ===")
    # Classic Josephus problem: 41 people, eliminate every 2nd
    survivor_position = solve_card_arrangement(
        41,
        "Arrange 41 people in a circle, eliminate every second person. What position survives?",
    )
    print(f"Survivor position: {survivor_position} - (expected 19)")

    print("\n=== Logic Grid ===")
    clues = [
        "There are 3 people: Alice, Bob, and Carol",
        "There are 3 house colors: red, blue, green",
        "Alice is older than Bob",
        "Carol is younger than Bob",
        "The oldest person lives in the red house",
        "Alice doesn't live in the blue house",
        "Carol lives in the green house",
    ]

    answer = solve_logic_grid(clues, "Who lives in the blue house?")
    print(f"Blue house resident: {answer} - (expected 'Bob')")


if __name__ == "__main__":
    main()
