"""
Benchmark for examples/funcy.py - Simple Function Generation
"""

from typing import Callable

from agex.bench import Trial, benchmark_pass_fail, params
from examples.funcy import fn_builder


def _equivalent(expected_fn: Callable, actual_fn: Callable) -> bool:
    test_inputs = range(8)  # 0-7, factorial of 7 = 5040 which is reasonable
    return all(expected_fn(x) == actual_fn(x) for x in test_inputs)


def _equivalent_pairs(expected_fn: Callable, actual_fn: Callable) -> bool:
    test_pairs = [(i, j) for i in range(5) for j in range(5)]
    return all(expected_fn(x, y) == actual_fn(x, y) for x, y in test_pairs)


def main():
    """Run the funcy benchmark."""

    # Reference implementations
    def factorial(n):
        """Reference factorial implementation."""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    # Define test cases using reference functions
    trials = [
        Trial(
            params=params("a function that returns the factorial of a number"),
            expected=factorial,
            judge=_equivalent,
        ),
        Trial(
            params=params("a function that checks if a number is even"),
            expected=lambda x: x % 2 == 0,
            judge=_equivalent,
        ),
        Trial(
            params=params("a function that returns the square of a number"),
            expected=lambda x: x * x,
            judge=_equivalent,
        ),
        Trial(
            params=params("a function that returns the absolute value of a number"),
            expected=lambda x: abs(x),
            judge=_equivalent,
        ),
        Trial(
            params=params("a function that returns the maximum of two numbers"),
            expected=lambda x, y: max(x, y),
            judge=_equivalent_pairs,
        ),
    ]

    # Run the benchmark
    print("Running funcy benchmark...")
    print("=" * 50)

    results = benchmark_pass_fail(
        tasks=[fn_builder],
        trials=trials,
        max_concurrency=5,
    )

    # Print results
    for task, stats in results.items():
        print(f"\nTask: {task}")
        print(f"Completed: {stats.completed_trials}/{stats.total_trials}")
        print(f"Passed: {stats.pass_count}/{stats.pass_count+stats.fail_count}")
        print(f"Average actions per trial: {stats.actions_per_trial:.1f}")
        print(f"Time per trial: {stats.time_per_trial:.2f} seconds")


if __name__ == "__main__":
    main()
