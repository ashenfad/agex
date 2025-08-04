"""
Benchmark for Excel Analysis - Data Processing and Business Intelligence
Tests agent's ability to analyze business data using pandas operations.
"""

import pandas as pd
from typing import Callable
import pytest
from IPython.display import display
from agex import Agent, connect_llm, Event, TaskStartEvent
from agex.bench import Trial, benchmark_pass_fail, params
from agex.helpers import register_pandas, register_stdlib

CASE1_EXPECTED = 1109.50
CASE2_EXPECTED = 1149.50
CASE3_EXPECTED = 502.00
CASE4_EXPECTED = {"Books": 0.0, "Clothing": 20.0, "Electronics": 13.0}
CASE5_EXPECTED = {"Books": 25.0, "Clothing": 40.0, "Electronics": 143.7}
CASE5_EXPECTED_ALT = {
    "Books": 25.0,
    "Clothing": 40.0,
    "Electronics": 123.27777777777777,
}


def make_task(model: str) -> Callable:
    """Create an excel analysis task with specified model."""
    llm_client = connect_llm(
        "openai",
        model=model,
        base_url=None if "gpt" in model else "http://localhost:11434/v1",
    )
    analyst = Agent(
        name=f"analyst-{model}",
        primer="You are an expert business data analyst skilled at pandas operations.",
        llm_client=llm_client,
        max_iterations=8,
    )
    register_pandas(analyst)
    register_stdlib(analyst)

    @analyst.task
    def analyze_sales_data(sales_df: pd.DataFrame, question: str) -> float | dict:  # type: ignore[return-value]
        """
        Analyze sales data and answer business questions.
        Return the result as requested (float for single values, dict for grouped results).
        """
        pass

    return analyze_sales_data


def create_benchmark_data() -> pd.DataFrame:
    """Load deterministic sample data from CSV for consistent benchmarking."""
    import os

    # Get the path to the CSV file relative to this script
    csv_path = os.path.join(os.path.dirname(__file__), "excel_analysis_data.csv")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def main():
    """Run the excel analysis benchmark."""

    # Create benchmark data and calculate expected results
    print("Creating benchmark data...")
    df = create_benchmark_data()

    print("Dataset info:")
    print(f"- Total records: {len(df)}")
    print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"- Categories: {df['Category'].unique().tolist()}")
    print(f"- Revenue range: ${df['Revenue'].min():.2f} to ${df['Revenue'].max():.2f}")

    def approx_equal_float(expected: float) -> Callable[[float], bool]:
        """Judge function for float results."""
        return lambda actual: bool(actual == pytest.approx(expected, abs=0.01))

    def approx_equal_dict(expected: dict) -> Callable[[dict], bool]:
        """Judge function for dictionary results."""
        return lambda actual: bool(actual == pytest.approx(expected, abs=0.01))

    # Define test cases with escalating complexity
    trials = [
        # Level 1: Basic filtering
        Trial(
            params=params(
                df,
                "Calculate the total revenue for all electronics category items",
                on_event=display,
            ),
            judge=approx_equal_float(CASE1_EXPECTED),
        ),
        # Level 2: Date filtering
        Trial(
            params=params(
                df,
                "Calculate the total revenue for all items sold in Q2 2023 (April, May, June)",
                on_event=display,
            ),
            judge=approx_equal_float(CASE2_EXPECTED),
        ),
        # # Level 3: Multi-condition logic
        Trial(
            params=params(
                df,
                "Calculate the total revenue from electronics category items sold in Q2 2023, excluding items with discount greater than 20%",
                on_event=display,
            ),
            judge=approx_equal_float(CASE3_EXPECTED),
        ),
        # Level 4: Aggregation
        Trial(
            params=params(
                df,
                "Calculate the average discount percentage by category. Return as a dictionary.",
                on_event=display,
            ),
            judge=approx_equal_dict(CASE4_EXPECTED),
        ),
        # Level 5: Complex calculation
        Trial(
            params=params(
                df,
                "Calculate the average revenue per unit by category. Return as a dictionary.",
                on_event=display,
            ),
            judge=lambda actual: any(
                approx_equal_dict(expected)(actual)
                for expected in (CASE5_EXPECTED, CASE5_EXPECTED_ALT)
            ),
        ),
    ]

    # Run the benchmark
    print("\nRunning excel analysis benchmark...")
    print("=" * 60)

    results = benchmark_pass_fail(
        tasks=[
            # make_task("qwen3:0.6b"),
            # make_task("qwen3:1.7b"),
            # make_task("qwen3:4b"),
            # make_task("qwen3:8b"),
            # make_task("qwen3:14b"),
            # make_task("qwen3-coder:30b"),
            make_task("gpt-4.1-nano"),
        ],
        trials=trials * 2,  # Run each test case 2 times
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
# ------------
# Local models
# ------------
#
# Task: <agex.task analyst-qwen3:0.6b/analyze_sales_data at 0x138755940>
# Completed trials: 3/10
# Passed trials: 0/3
# Actions per trial: 3.7
#
# Task: <agex.task analyst-qwen3:1.7b/analyze_sales_data at 0x130744e30>
# Completed trials: 7/10
# Passed trials: 3/7
# Actions per trial: 3.0
#
# Task: <agex.task analyst-qwen3:4b/analyze_sales_data at 0x134ae6b10>
# Completed trials: 7/10
# Passed trials: 5/7
# Actions per trial: 2.6
#
# Task: <agex.task analyst-qwen3:8b/analyze_sales_data at 0x132f45160>
# Completed trials: 9/10
# Passed trials: 9/9
# Actions per trial: 3.0
#
# Task: <agex.task analyst-qwen3:14b/analyze_sales_data at 0x116e11220>
# Completed trials: 10/10
# Passed trials: 8/10
# Actions per trial: 3.3
#
# Task: <agex.task analyst-qwen3-coder:30b/analyze_sales_data at 0x140b0da60>
# Completed trials: 10/10
# Passed trials: 10/10
# Actions per trial: 3.1
#
# -------------
# Remote models
# -------------
#
# Task: <agex.task analyst-gpt-4.1-nano/analyze_sales_data at 0x143c92f90>
# Completed trials: 10/10
# Passed trials: 10/10
# Actions per trial: 1.8
