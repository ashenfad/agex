"""
agex.bench - Native benchmarking framework for agex agents.

This module provides tools for empirically testing and comparing
agent performance across different configurations, primers, and models.

Key Features:
- Works with Python objects (not just text)
- Agent tasks as judges ("LLM-as-judge" with full agex capabilities)
- Automatic event-based performance metrics
- Type-safe generic API with simple convenience functions
- Concurrent execution support

Basic Usage:
    >>> from agex.bench import benchmark_pass_fail, Trial, params
    >>> import operator
    >>>
    >>> # Define test trials
    >>> trials = [
    ...     Trial(params("What is 2+2?"), expected="4", judge=operator.eq),
    ...     Trial(params("What is 3+3?"), expected="6", judge=operator.eq),
    ... ]
    >>>
    >>> # Benchmark agents
    >>> results = benchmark_pass_fail([agent1.task, agent2.task], trials)
    >>>
    >>> # View results
    >>> for task, stats in results.items():
    ...     print(f"{task}: {stats.pass_rate:.1%} success rate")

Numeric Benchmarking:
    >>> from agex.bench import benchmark_numeric, Trial, params
    >>>
    >>> def quality_judge(expected, actual):
    ...     return len(actual) / 10.0  # Simple quality score
    >>>
    >>> trials = [Trial(params("Write a story"), expected="good", judge=quality_judge)]
    >>> results = benchmark_numeric([writer.task], trials)
    >>>
    >>> for task, stats in results.items():
    ...     print(f"{task}: avg={stats.mean_score:.1f}, range={stats.min_score:.1f}-{stats.max_score:.1f}")

Advanced Usage:
    >>> from agex.bench import benchmark_generic, Trial, params
    >>>
    >>> def custom_judge(expected, actual):
    ...     return {"accuracy": expected == actual, "fluency": score_fluency(actual)}
    >>>
    >>> def custom_aggregator(results, event_stats):
    ...     return CustomStats(
    ...         avg_accuracy=mean(r["accuracy"] for r in results),
    ...         avg_fluency=mean(r["fluency"] for r in results)
    ...     )
    >>>
    >>> trials = [Trial(params("Write a story"), expected="good_story", judge=custom_judge)]
    >>> results = benchmark_generic([writer1.task, writer2.task], trials, custom_aggregator)
"""

from .aggregators import numeric_aggregator, pass_fail_aggregator
from .core import benchmark_generic, benchmark_numeric, benchmark_pass_fail
from .types import NumericStats, Params, PassFailStats, Stats, Trial, params

__all__ = [
    # Core benchmark functions
    "benchmark_generic",
    "benchmark_pass_fail",
    "benchmark_numeric",
    # Types
    "Trial",
    "Params",
    "params",
    "Stats",
    "PassFailStats",
    "NumericStats",
    # Built-in aggregators
    "pass_fail_aggregator",
    "numeric_aggregator",
]
