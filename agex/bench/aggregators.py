"""
Built-in aggregation functions for agex.bench.

This module provides common aggregation strategies for converting
lists of judge results into Stats objects.
"""

from .types import NumericStats, PassFailStats, Stats


def pass_fail_aggregator(results: list[bool], event_stats: Stats) -> PassFailStats:
    """Aggregate boolean results into PassFailStats.

    Args:
        results: List of boolean results from judge functions
        event_stats: Stats object with event-derived metrics

    Returns:
        PassFailStats with aggregated metrics

    Raises:
        ValueError: If any result is not a boolean

    Example:
        >>> results = [True, False, True, True]
        >>> event_stats = Stats(total_trials=4, completed_trials=4, ...)
        >>> stats = pass_fail_aggregator(results, event_stats)
        >>> stats.pass_rate
        0.75
    """
    if not all(isinstance(r, bool) for r in results):
        invalid_types = {type(r).__name__ for r in results if not isinstance(r, bool)}
        raise ValueError(
            f"pass_fail_aggregator requires boolean results from judge functions. "
            f"Found non-boolean types: {invalid_types}"
        )

    # results only contains results from completed trials (errored trials were skipped)
    pass_count = sum(results)
    fail_count = len(results) - pass_count

    return PassFailStats(
        total_trials=event_stats.total_trials,
        completed_trials=event_stats.completed_trials,
        errored_trials=event_stats.errored_trials,
        actions_per_trial=event_stats.actions_per_trial,
        time_per_trial=event_stats.time_per_trial,
        pass_count=pass_count,
        fail_count=fail_count,
    )


def numeric_aggregator(results: list[float], event_stats: Stats) -> NumericStats:
    """Aggregate numeric results into NumericStats.

    Args:
        results: List of numeric results from judge functions
        event_stats: Stats object with event-derived metrics

    Returns:
        NumericStats with aggregated metrics and numeric aggregations

    Raises:
        ValueError: If any result is not numeric
    """
    if not all(isinstance(r, (int, float)) for r in results):
        invalid_types = {
            type(r).__name__ for r in results if not isinstance(r, (int, float))
        }
        raise ValueError(
            f"numeric_aggregator requires numeric results from judge functions. "
            f"Found non-numeric types: {invalid_types}"
        )

    # If no trials were successful, there are no results to aggregate
    # Return zeroed-out stats in this case
    if not results:
        return NumericStats(
            # Base stats from events
            total_trials=event_stats.total_trials,
            completed_trials=event_stats.completed_trials,
            errored_trials=event_stats.errored_trials,
            actions_per_trial=event_stats.actions_per_trial,
            time_per_trial=event_stats.time_per_trial,
            # Zeroed-out numeric aggregations
            mean_score=0.0,
            min_score=0.0,
            max_score=0.0,
            total_score=0.0,
        )

    # Compute numeric aggregations
    mean_score = sum(results) / len(results)
    min_score = min(results)
    max_score = max(results)
    total_score = sum(results)

    return NumericStats(
        # Base stats from events
        total_trials=event_stats.total_trials,
        completed_trials=event_stats.completed_trials,
        errored_trials=event_stats.errored_trials,
        actions_per_trial=event_stats.actions_per_trial,
        time_per_trial=event_stats.time_per_trial,
        # Numeric aggregations
        mean_score=mean_score,
        min_score=min_score,
        max_score=max_score,
        total_score=total_score,
    )
