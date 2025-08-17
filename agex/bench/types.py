"""
Core types for the agex.bench benchmarking framework.

This module provides the fundamental data structures used for defining
benchmark trials, parameters, and results.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

# Type variables for generic trial definitions
T = TypeVar("T")  # Type of expected/actual values
U = TypeVar("U")  # Type of judge function result


@dataclass
class Params:
    """Container for task function parameters."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def params(*args, **kwargs) -> Params:
    """Convenience function to create Params instances.

    Args:
        *args: Positional arguments to pass to task function
        **kwargs: Keyword arguments to pass to task function

    Returns:
        Params instance containing the arguments

    Example:
        >>> p = params("hello", count=5)
        >>> p.args
        ('hello',)
        >>> p.kwargs
        {'count': 5}
    """
    return Params(args, kwargs)


@dataclass
class Trial(Generic[T, U]):
    """A single benchmark trial definition.

    Represents one test case with input parameters and a judge function
    to evaluate the actual result.

    Type Parameters:
        T: Type of the actual value
        U: Type of judge function result

    Attributes:
        params: Input parameters to pass to task function
        judge: Function that evaluates the actual result and returns another value
    """

    params: Params
    judge: Callable[[T], U]


@dataclass
class Stats:
    """Base class for benchmark statistics.

    Contains metrics automatically collected from agex's event system
    during benchmark execution.
    """

    total_trials: int
    completed_trials: int  # Number of trials that completed without errors
    errored_trials: int  # Number of trials that failed with errors
    actions_per_trial: float  # Average LLM calls per completed trial
    time_per_trial: float  # Average execution time per completed trial


@dataclass
class PassFailStats(Stats):
    """Statistics for pass/fail benchmark results.

    Extends base Stats with pass/fail metrics for boolean evaluations.
    """

    pass_count: int  # Number of completed trials that passed
    fail_count: int  # Number of completed trials that failed

    @property
    def pass_rate(self) -> float:
        """Percentage of completed trials that passed (0.0 to 1.0)."""
        total = self.pass_count + self.fail_count
        return self.pass_count / total if total > 0 else 0.0


@dataclass
class NumericStats(Stats):
    """Statistics for numeric benchmark results.

    Extends base Stats with numeric aggregations of judge results.
    """

    mean_score: float  # Average of all numeric results
    min_score: float  # Minimum numeric result
    max_score: float  # Maximum numeric result
    total_score: float  # Sum of all numeric results
