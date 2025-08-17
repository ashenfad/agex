"""
Benchmark for examples/funcy.py - Simple Function Generation
"""

import math
from typing import Callable

from IPython.display import display

from agex import Agent, connect_llm
from agex.bench import Trial, benchmark_pass_fail, params


def make_task(model: str) -> Callable:
    llm_client = connect_llm(
        "openai",
        model=model,
        base_url="http://localhost:11434/v1",
        temperature=0.1,
    )

    funcy = Agent(
        name=f"funcy-{model}",
        primer="You are great at providing custom functions to the user.",
        llm_client=llm_client,
        max_iterations=4,
    )
    funcy.module(math, visibility="low")

    @funcy.task
    def fn_builder(prompt: str) -> Callable:  # type: ignore[return-value]
        """
        Build a callable function from a text prompt. Return the function w/ task_success.
        """
        pass

    return fn_builder


def main():
    """Run the funcy benchmark."""

    def factorial(n):
        """Reference factorial implementation."""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def equivalent(expected_fn: Callable) -> Callable[[Callable], bool]:
        """Creates a judge function to check for single-argument function equivalence."""

        def judge(actual_fn: Callable) -> bool:
            test_inputs = range(8)
            try:
                return all(expected_fn(x) == actual_fn(x) for x in test_inputs)
            except Exception:
                return False

        return judge

    def equivalent_pairs(expected_fn: Callable) -> Callable[[Callable], bool]:
        """Creates a judge function to check for two-argument function equivalence."""

        def judge(actual_fn: Callable) -> bool:
            test_pairs = [(i, j) for i in range(5) for j in range(5)]
            try:
                return all(expected_fn(x, y) == actual_fn(x, y) for x, y in test_pairs)
            except Exception:
                return False

        return judge

    # Define test cases using reference functions
    trials = [
        Trial(
            params=params(
                "a fn that returns the factorial of a given number", on_event=display
            ),
            judge=equivalent(factorial),
        ),
        Trial(
            params=params(
                "a fn that checks if a given number is even", on_event=display
            ),
            judge=equivalent(lambda x: x % 2 == 0),
        ),
        Trial(
            params=params(
                "a fn that returns the square of a given number", on_event=display
            ),
            judge=equivalent(lambda x: x * x),
        ),
        Trial(
            params=params(
                "a fn that returns the absolute value of a given number",
                on_event=display,
            ),
            judge=equivalent(abs),
        ),
        Trial(
            params=params(
                "a fn that returns the maximum of two given numbers", on_event=display
            ),
            judge=equivalent_pairs(lambda x, y: max(x, y)),
        ),
    ]

    # Run the benchmark
    print("Running funcy benchmark...")
    print("=" * 50)

    results = benchmark_pass_fail(
        tasks=[
            make_task("qwen3:0.6b"),
            make_task("qwen3:1.7b"),
            make_task("qwen3:4b"),
        ],
        trials=trials * 3,
        max_concurrency=1,
    )

    # Print results
    for task, stats in results.items():
        print(f"\nTask: {task}")
        print(f"Completed trials: {stats.completed_trials}/{stats.total_trials}")
        print(
            f"Passed trials: {stats.pass_count}/{stats.pass_count + stats.fail_count}"
        )
        print(f"Actions per trial: {stats.actions_per_trial:.1f}")
        print(f"Time per trial: {stats.time_per_trial:.2f} seconds")


if __name__ == "__main__":
    main()


# Task: <agex.task funcy-qwen3:0.6b/fn_builder at 0x117a18200>
# Completed trials: 11/15
# Passed trials: 11/11
# Actions per trial: 1.0
# Time per trial: 2.38 seconds

# Task: <agex.task funcy-qwen3:1.7b/fn_builder at 0x117a18f50>
# Completed trials: 15/15
# Passed trials: 15/15
# Actions per trial: 1.0
# Time per trial: 2.78 seconds

# Task: <agex.task funcy-qwen3:4b/fn_builder at 0x117a19ca0>
# Completed trials: 15/15
# Passed trials: 15/15
# Actions per trial: 1.0
# Time per trial: 8.06 seconds
