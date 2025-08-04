"""
Tests for agex.bench.aggregators module.

Tests for aggregation functions: pass_fail_aggregator and numeric_aggregator.
"""

import pytest

from agex.bench.aggregators import numeric_aggregator, pass_fail_aggregator
from agex.bench.types import NumericStats, PassFailStats, Stats


def test_pass_fail_aggregator_basic():
    """Test basic pass_fail_aggregator functionality."""
    # Create mock event stats
    event_stats = Stats(
        total_trials=4,
        completed_trials=4,
        errored_trials=0,
        actions_per_trial=2.0,
        time_per_trial=8.5,
    )

    # Test with mixed results
    results = [True, False, True, True]
    stats = pass_fail_aggregator(results, event_stats)

    # Verify it returns PassFailStats
    assert isinstance(stats, PassFailStats)

    # Verify base stats are preserved
    assert stats.total_trials == 4
    assert stats.completed_trials == 4
    assert stats.errored_trials == 0
    assert stats.actions_per_trial == 2.0
    assert stats.time_per_trial == 8.5

    # Verify pass/fail calculations
    assert stats.pass_count == 3
    assert stats.fail_count == 1
    assert abs(stats.pass_rate - 0.75) < 0.001  # 3/4 = 0.75


def test_pass_fail_aggregator_edge_cases():
    """Test edge cases for pass_fail_aggregator."""
    event_stats = Stats(
        total_trials=2,
        completed_trials=2,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=2.0,
    )

    # All pass
    all_pass = pass_fail_aggregator([True, True], event_stats)
    assert all_pass.pass_count == 2
    assert all_pass.fail_count == 0
    assert all_pass.pass_rate == 1.0

    # All fail
    all_fail = pass_fail_aggregator([False, False], event_stats)
    assert all_fail.pass_count == 0
    assert all_fail.fail_count == 2
    assert all_fail.pass_rate == 0.0

    # Single result
    single_pass = pass_fail_aggregator([True], event_stats)
    assert single_pass.pass_count == 1
    assert single_pass.fail_count == 0
    assert single_pass.pass_rate == 1.0


def test_pass_fail_aggregator_invalid_input():
    """Test pass_fail_aggregator with invalid input."""
    event_stats = Stats(
        total_trials=3,
        completed_trials=3,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=3.0,
    )

    # Test with non-boolean results
    with pytest.raises(
        ValueError, match="pass_fail_aggregator requires boolean results"
    ):
        pass_fail_aggregator([True, "not_bool", False], event_stats)

    with pytest.raises(
        ValueError, match="pass_fail_aggregator requires boolean results"
    ):
        pass_fail_aggregator([1, 0], event_stats)

    # Test with empty results
    empty_stats = pass_fail_aggregator([], event_stats)
    assert empty_stats.pass_count == 0
    assert empty_stats.fail_count == 0
    assert empty_stats.pass_rate == 0.0


def test_numeric_aggregator_basic():
    """Test basic numeric_aggregator functionality."""
    # Create mock event stats
    event_stats = Stats(
        total_trials=4,
        completed_trials=4,
        errored_trials=0,
        actions_per_trial=2.5,
        time_per_trial=10.2,
    )

    # Test with numeric results
    results = [1.5, 3.2, 2.1, 4.8]
    stats = numeric_aggregator(results, event_stats)

    # Verify it returns NumericStats
    assert isinstance(stats, NumericStats)

    # Verify base stats are preserved
    assert stats.total_trials == 4
    assert stats.completed_trials == 4
    assert stats.errored_trials == 0
    assert stats.actions_per_trial == 2.5
    assert stats.time_per_trial == 10.2

    # Verify numeric calculations
    expected_mean = sum(results) / len(results)  # (1.5 + 3.2 + 2.1 + 4.8) / 4 = 2.9
    expected_min = min(results)  # 1.5
    expected_max = max(results)  # 4.8
    expected_total = sum(results)  # 11.6

    assert abs(stats.mean_score - expected_mean) < 0.001
    assert abs(stats.min_score - expected_min) < 0.001
    assert abs(stats.max_score - expected_max) < 0.001
    assert abs(stats.total_score - expected_total) < 0.001


def test_numeric_aggregator_edge_cases():
    """Test edge cases for numeric_aggregator."""
    event_stats = Stats(
        total_trials=1,
        completed_trials=1,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=1.0,
    )

    # Single value
    single_result = numeric_aggregator([5.5], event_stats)
    assert single_result.mean_score == 5.5
    assert single_result.min_score == 5.5
    assert single_result.max_score == 5.5
    assert single_result.total_score == 5.5

    # All same values
    same_values = numeric_aggregator([3.0, 3.0, 3.0], event_stats)
    assert same_values.mean_score == 3.0
    assert same_values.min_score == 3.0
    assert same_values.max_score == 3.0
    assert same_values.total_score == 9.0

    # Mix of integers and floats
    mixed_numbers = numeric_aggregator([1, 2.5, 3], event_stats)
    expected_mean = (1 + 2.5 + 3) / 3
    assert abs(mixed_numbers.mean_score - expected_mean) < 0.001
    assert mixed_numbers.min_score == 1
    assert mixed_numbers.max_score == 3
    assert mixed_numbers.total_score == 6.5

    # Negative numbers
    negative_numbers = numeric_aggregator([-1.0, 2.0, -0.5], event_stats)
    assert abs(negative_numbers.mean_score - (0.5 / 3)) < 0.001
    assert negative_numbers.min_score == -1.0
    assert negative_numbers.max_score == 2.0
    assert negative_numbers.total_score == 0.5


def test_numeric_aggregator_invalid_input():
    """Test numeric_aggregator with invalid input."""
    event_stats = Stats(
        total_trials=3,
        completed_trials=3,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=3.0,
    )

    # Test with non-numeric results
    with pytest.raises(ValueError, match="numeric_aggregator requires numeric results"):
        numeric_aggregator([1.0, "not_number", 3.0], event_stats)

    # Note: booleans are considered numeric in Python (bool is subclass of int)
    # So [True, False] would actually work and return [1.0, 0.0] scores
    bool_stats = numeric_aggregator([True, False], event_stats)
    assert bool_stats.mean_score == 0.5  # (1 + 0) / 2

    # Test with empty results
    empty_stats = numeric_aggregator([], event_stats)
    assert empty_stats.mean_score == 0.0
    assert empty_stats.min_score == 0.0
    assert empty_stats.max_score == 0.0
    assert empty_stats.total_score == 0.0


def test_aggregator_type_consistency():
    """Test that aggregators maintain type consistency."""
    event_stats = Stats(
        total_trials=2,
        completed_trials=2,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=2.0,
    )

    # Pass/fail aggregator should always return PassFailStats
    pf_stats = pass_fail_aggregator([True, False], event_stats)
    assert type(pf_stats) == PassFailStats
    assert isinstance(pf_stats, Stats)  # Should also be a Stats

    # Numeric aggregator should always return NumericStats
    num_stats = numeric_aggregator([1.0, 2.0], event_stats)
    assert type(num_stats) == NumericStats
    assert isinstance(num_stats, Stats)  # Should also be a Stats

    # They should be different types
    assert type(pf_stats) != type(num_stats)
