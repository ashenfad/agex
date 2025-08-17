"""
Core benchmarking functions for agex.bench.

This module provides the main benchmark execution functions that run
task functions against test trials and collect performance metrics.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, cast

from agex.agent.events import ActionEvent, BaseEvent

from .aggregators import numeric_aggregator, pass_fail_aggregator
from .types import NumericStats, PassFailStats, Stats, Trial

# Type variables
T = TypeVar("T")
U = TypeVar("U")


@dataclass
class TrialResult(Generic[T, U]):
    """Result of running a single trial."""

    trial: Trial[T, U]
    result: Any  # Actual result from task execution
    events: list[BaseEvent]  # Events collected during execution
    error: Exception | None  # Error if trial failed, None if successful

    @property
    def succeeded(self) -> bool:
        """True if the trial completed without error."""
        return self.error is None


def benchmark_generic(
    tasks: list[Callable[..., T]],
    trials: list[Trial[T, U]],
    agg: Callable[[list[U], Stats], Stats],
    max_concurrency: int = 1,
) -> dict[Callable, Stats]:
    """Generic benchmark function for flexible evaluation scenarios.

    Runs a list of task functions against a set of trials and aggregates
    the results using a custom aggregation function. Supports any judge
    function return type and custom statistics.

    Args:
        tasks: List of task functions to benchmark
        trials: List of trial definitions to run against each task
        agg: Aggregation function that converts judge results to Stats
        max_concurrency: Maximum number of trials to run concurrently

    Returns:
        Dictionary mapping each task function to its aggregated Stats

    Raises:
        ValueError: If tasks or trials lists are empty
        TypeError: If judge function signatures don't match expected/actual types

    Example:
        >>> def custom_judge(expected, actual):
        ...     return {"accuracy": expected == actual, "length": len(str(actual))}
        >>>
        >>> def custom_agg(results, event_stats):
        ...     return CustomStats(avg_accuracy=mean(r["accuracy"] for r in results))
        >>>
        >>> results = benchmark_generic([task1, task2], trials, custom_agg)
    """
    if not tasks:
        raise ValueError("Cannot benchmark empty task list")
    if not trials:
        raise ValueError("Cannot benchmark with empty trials list")

    results = {}

    for task in tasks:
        task_results = _run_task_trials(task, trials, max_concurrency)
        judge_results = []
        event_stats = _compute_event_stats(task_results)

        for trial_result in task_results:
            if not trial_result.succeeded:
                # For now, skip failed trials - could make configurable later
                continue

            try:
                judge_result = trial_result.trial.judge(trial_result.result)
                judge_results.append(judge_result)
            except Exception as e:
                raise TypeError(
                    f"Judge function failed for trial. "
                    f"actual={trial_result.result}. Error: {e}"
                ) from e

        try:
            aggregated_stats = agg(judge_results, event_stats)
            results[task] = aggregated_stats
        except Exception as e:
            raise ValueError(
                f"Aggregation failed for task {task}. "
                f"Judge results: {judge_results[:3]}... Error: {e}"
            ) from e

    return results


def benchmark_pass_fail(
    tasks: list[Callable[..., T]],
    trials: list[Trial[T, bool]],
    max_concurrency: int = 1,
) -> dict[Callable[..., T], PassFailStats]:
    """Simple pass/fail benchmark for boolean evaluations.

    Convenience wrapper around benchmark_generic for the common case
    of boolean judge functions and pass/fail statistics.

    Args:
        tasks: List of task functions to benchmark
        trials: List of trial definitions with boolean judge functions
        max_concurrency: Maximum number of trials to run concurrently

    Returns:
        Dictionary mapping each task function to PassFailStats

    Example:
        >>> import operator
        >>> trials = [
        ...     Trial(params("What is 2+2?"), expected="4", judge=operator.eq),
        ...     Trial(params("What is 3+3?"), expected="6", judge=operator.eq),
        ... ]
        >>> results = benchmark_pass_fail([math_agent.task], trials)
        >>> results[math_agent.task].pass_rate
        1.0
    """
    # Cast to PassFailStats since we know pass_fail_aggregator returns PassFailStats
    return cast(
        dict[Callable[..., T], PassFailStats],
        benchmark_generic(tasks, trials, pass_fail_aggregator, max_concurrency),
    )


def benchmark_numeric(
    tasks: list[Callable[..., T]],
    trials: list[Trial[T, float]],
    max_concurrency: int = 1,
) -> dict[Callable[..., T], NumericStats]:
    """Simple numeric benchmark for numeric evaluations.

    Convenience wrapper around benchmark_generic for the common case
    of numeric judge functions and numeric statistics.

    Args:
        tasks: List of task functions to benchmark
        trials: List of trial definitions with numeric judge functions
        max_concurrency: Maximum number of trials to run concurrently

    Returns:
        Dictionary mapping each task function to NumericStats

    Example:
        >>> def quality_judge(expected, actual):
        ...     return len(actual) / 10.0  # Simple quality score
        >>>
        >>> trials = [
        ...     Trial(params("Write a story"), expected="good", judge=quality_judge),
        ...     Trial(params("Write a poem"), expected="good", judge=quality_judge),
        ... ]
        >>> results = benchmark_numeric([writer_agent.task], trials)
        >>> results[writer_agent.task].mean_score
        4.2
    """
    # Cast to NumericStats since we know numeric_aggregator returns NumericStats
    return cast(
        dict[Callable[..., T], NumericStats],
        benchmark_generic(tasks, trials, numeric_aggregator, max_concurrency),
    )


def _run_task_trials(
    task: Callable[..., T], trials: list[Trial[T, U]], max_concurrency: int
) -> list[TrialResult[T, U]]:
    """Run a single task against all trials, collecting events and handling errors.

    Returns:
        List of TrialResult objects for each trial
    """
    if max_concurrency == 1:
        # Sequential execution
        return [_run_single_trial(task, trial) for trial in trials]
    else:
        # Concurrent execution
        results = []
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            # Submit all trials with their indices
            future_to_index = {
                executor.submit(_run_single_trial, task, trial): i
                for i, trial in enumerate(trials)
            }

            # Collect results using indices
            trial_results = {}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                trial_results[index] = future.result()

            # Return results in original trial order
            results = [trial_results[i] for i in range(len(trials))]

        return results


def _run_single_trial(task: Callable[..., T], trial: Trial[T, U]) -> TrialResult[T, U]:
    """Run a single trial and collect events.

    Returns:
        TrialResult containing trial, result, events, and error info
    """
    events = []

    def event_collector(event: BaseEvent) -> None:
        events.append(event)

    kwargs = trial.params.kwargs.copy()

    try:
        # Execute task with event collection
        if on_event := trial.params.kwargs.get("on_event"):

            def handler(event: BaseEvent):
                event_collector(event)
                on_event(event)

            del kwargs["on_event"]

        else:
            handler = event_collector
        result = task(*trial.params.args, on_event=handler, **kwargs)
        return TrialResult(trial=trial, result=result, events=events, error=None)
    except Exception as e:
        return TrialResult(trial=trial, result=None, events=events, error=e)


def _compute_event_stats(
    task_results: list[TrialResult],
) -> Stats:
    """Compute statistics from collected events.

    Args:
        task_results: List of TrialResult objects

    Returns:
        Dictionary of aggregated event statistics
    """
    successful_trials = [tr for tr in task_results if tr.succeeded]
    error_count = sum(1 for tr in task_results if not tr.succeeded)

    total_trials = len(task_results)
    success_count = len(successful_trials)

    if not successful_trials:
        return Stats(
            total_trials=total_trials,
            completed_trials=0,
            errored_trials=error_count,
            actions_per_trial=0.0,
            time_per_trial=0.0,
        )

    # Count actions (LLM calls) and time for completed trials only
    total_actions = 0
    total_time = 0.0

    for trial_result in successful_trials:
        # Count action events (events that represent LLM calls)
        action_count = len(
            [e for e in trial_result.events if isinstance(e, ActionEvent)]
        )
        total_actions += action_count

        # Calculate trial time from first to last event
        if trial_result.events:
            trial_start = min(e.timestamp for e in trial_result.events)
            trial_end = max(e.timestamp for e in trial_result.events)
            time_diff = trial_end - trial_start
            total_time += time_diff.total_seconds()

    # Calculate averages for completed trials
    actions_per_trial = total_actions / success_count if success_count > 0 else 0.0
    time_per_trial = total_time / success_count if success_count > 0 else 0.0

    return Stats(
        total_trials=total_trials,
        completed_trials=success_count,
        errored_trials=error_count,
        actions_per_trial=actions_per_trial,
        time_per_trial=time_per_trial,
    )
