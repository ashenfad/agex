"""
Tests for agex.bench.types module.

Tests for core data structures: Params, Trial, Stats, PassFailStats, NumericStats,
and helper functions like params().
"""

from agex.bench.types import NumericStats, Params, PassFailStats, Stats, Trial, params


def test_params_creation():
    """Test Params dataclass creation and params() helper."""
    # Test with positional args only
    p1 = params("hello", "world")
    assert p1.args == ("hello", "world")
    assert p1.kwargs == {}

    # Test with keyword args only
    p2 = params(name="test", count=5)
    assert p2.args == ()
    assert p2.kwargs == {"name": "test", "count": 5}

    # Test with mixed args
    p3 = params("hello", count=5, flag=True)
    assert p3.args == ("hello",)
    assert p3.kwargs == {"count": 5, "flag": True}

    # Test empty params
    p4 = params()
    assert p4.args == ()
    assert p4.kwargs == {}

    # Test direct Params creation
    p5 = Params(("a", "b"), {"x": 1})
    assert p5.args == ("a", "b")
    assert p5.kwargs == {"x": 1}


def test_trial_creation():
    """Test Trial dataclass creation."""
    # Simple trial
    trial = Trial(params=params("test"), judge=lambda actual: actual == "result")

    assert trial.params.args == ("test",)
    assert trial.judge("result")
    assert not trial.judge("different")

    # Trial with complex params
    trial2 = Trial(
        params=params("input", count=3, flag=True),
        judge=lambda actual: actual == 42,
    )

    assert trial2.params.args == ("input",)
    assert trial2.params.kwargs == {"count": 3, "flag": True}
    assert trial2.judge(42)


def test_stats_dataclass():
    """Test base Stats dataclass."""
    stats = Stats(
        total_trials=10,
        completed_trials=8,
        errored_trials=2,
        actions_per_trial=2.5,
        time_per_trial=15.3,
    )

    assert stats.total_trials == 10
    assert stats.completed_trials == 8
    assert stats.errored_trials == 2
    assert stats.actions_per_trial == 2.5
    assert stats.time_per_trial == 15.3


def test_pass_fail_stats():
    """Test PassFailStats dataclass and pass_rate property."""
    stats = PassFailStats(
        total_trials=10,
        completed_trials=8,
        errored_trials=2,
        actions_per_trial=2.5,
        time_per_trial=15.3,
        pass_count=6,
        fail_count=2,
    )

    # Verify base Stats fields
    assert stats.total_trials == 10
    assert stats.completed_trials == 8
    assert stats.errored_trials == 2

    # Verify PassFailStats specific fields
    assert stats.pass_count == 6
    assert stats.fail_count == 2

    # Verify pass_rate property calculation
    expected_rate = 6 / (6 + 2)  # 6/8 = 0.75
    assert abs(stats.pass_rate - expected_rate) < 0.001
    assert abs(stats.pass_rate - 0.75) < 0.001

    # Test edge case: no trials
    empty_stats = PassFailStats(
        total_trials=0,
        completed_trials=0,
        errored_trials=0,
        actions_per_trial=0.0,
        time_per_trial=0.0,
        pass_count=0,
        fail_count=0,
    )
    assert empty_stats.pass_rate == 0.0

    # Test edge case: all pass
    all_pass = PassFailStats(
        total_trials=5,
        completed_trials=5,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=5.0,
        pass_count=5,
        fail_count=0,
    )
    assert all_pass.pass_rate == 1.0

    # Test edge case: all fail
    all_fail = PassFailStats(
        total_trials=3,
        completed_trials=3,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=3.0,
        pass_count=0,
        fail_count=3,
    )
    assert all_fail.pass_rate == 0.0


def test_numeric_stats():
    """Test NumericStats dataclass."""
    stats = NumericStats(
        total_trials=5,
        completed_trials=5,
        errored_trials=0,
        actions_per_trial=1.8,
        time_per_trial=12.5,
        mean_score=7.2,
        min_score=3.1,
        max_score=9.8,
        total_score=36.0,
    )

    # Verify base Stats fields
    assert stats.total_trials == 5
    assert stats.completed_trials == 5
    assert stats.errored_trials == 0
    assert stats.actions_per_trial == 1.8
    assert stats.time_per_trial == 12.5

    # Verify NumericStats specific fields
    assert stats.mean_score == 7.2
    assert stats.min_score == 3.1
    assert stats.max_score == 9.8
    assert stats.total_score == 36.0


def test_inheritance():
    """Test that PassFailStats and NumericStats properly inherit from Stats."""
    # Test isinstance relationships
    pass_fail_stats = PassFailStats(
        total_trials=1,
        completed_trials=1,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=1.0,
        pass_count=1,
        fail_count=0,
    )

    numeric_stats = NumericStats(
        total_trials=1,
        completed_trials=1,
        errored_trials=0,
        actions_per_trial=1.0,
        time_per_trial=1.0,
        mean_score=5.0,
        min_score=5.0,
        max_score=5.0,
        total_score=5.0,
    )

    # Both should be instances of Stats
    assert isinstance(pass_fail_stats, Stats)
    assert isinstance(numeric_stats, Stats)

    # And their specific types
    assert isinstance(pass_fail_stats, PassFailStats)
    assert isinstance(numeric_stats, NumericStats)

    # But not each other
    assert not isinstance(pass_fail_stats, NumericStats)
    assert not isinstance(numeric_stats, PassFailStats)
