"""
Tests for agex.bench.core module.

Tests for benchmark functions: benchmark_generic, benchmark_pass_fail, benchmark_numeric,
and internal functions.
"""

import operator
from unittest.mock import Mock

import pytest

from agex import Agent, clear_agent_registry
from agex.bench.core import (
    TrialResult,
    benchmark_generic,
    benchmark_numeric,
    benchmark_pass_fail,
)
from agex.bench.types import NumericStats, PassFailStats, Stats, Trial, params
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import DummyLLMClient


class TestTrialResult:
    """Test TrialResult dataclass."""

    def test_trial_result_creation(self):
        """Test TrialResult creation and properties."""
        trial = Trial(params("input"), expected="output", judge=operator.eq)
        events = []

        # Successful trial
        success_result = TrialResult(
            trial=trial, result="output", events=events, error=None
        )

        assert success_result.trial == trial
        assert success_result.result == "output"
        assert success_result.events == events
        assert success_result.error is None
        assert success_result.succeeded

        # Failed trial
        error = Exception("Something went wrong")
        failed_result = TrialResult(
            trial=trial, result=None, events=events, error=error
        )

        assert failed_result.trial == trial
        assert failed_result.result is None
        assert failed_result.events == events
        assert failed_result.error == error
        assert not failed_result.succeeded


class TestBenchmarkPassFail:
    """Test benchmark_pass_fail function."""

    def test_benchmark_pass_fail_basic(self):
        """Test basic pass/fail benchmarking."""
        clear_agent_registry()

        # Create agent with dummy responses
        dummy_responses = [
            LLMResponse(thinking="Solving 2+2", code="task_success('4')"),
            LLMResponse(thinking="Solving 1+1", code="task_success('2')"),
        ]
        client = DummyLLMClient(responses=dummy_responses)

        agent = Agent(name="math_agent")
        agent.llm_client = client

        @agent.task
        def solve_math(question: str) -> str:
            """Solve math problems."""
            pass

        # Create trials
        trials = [
            Trial(params("What is 2+2?"), expected="4", judge=operator.eq),
            Trial(params("What is 1+1?"), expected="2", judge=operator.eq),
        ]

        # Run benchmark
        results = benchmark_pass_fail([solve_math], trials)

        # Verify results
        assert len(results) == 1
        assert solve_math in results

        stats = results[solve_math]
        assert isinstance(stats, PassFailStats)
        assert stats.total_trials == 2
        assert stats.completed_trials == 2
        assert stats.errored_trials == 0
        assert stats.pass_count == 2
        assert stats.fail_count == 0
        assert stats.pass_rate == 1.0

    def test_benchmark_pass_fail_mixed_results(self):
        """Test pass/fail with some failures."""
        clear_agent_registry()

        # Responses where second one is wrong
        dummy_responses = [
            LLMResponse(thinking="Solving 2+2", code="task_success('4')"),  # Correct
            LLMResponse(thinking="Solving 1+1", code="task_success('3')"),  # Wrong!
        ]
        client = DummyLLMClient(responses=dummy_responses)

        agent = Agent(name="math_agent")
        agent.llm_client = client

        @agent.task
        def solve_math(question: str) -> str:
            """Solve math problems."""
            pass

        trials = [
            Trial(params("What is 2+2?"), expected="4", judge=operator.eq),
            Trial(params("What is 1+1?"), expected="2", judge=operator.eq),  # Will fail
        ]

        results = benchmark_pass_fail([solve_math], trials)
        stats = results[solve_math]

        assert stats.pass_count == 1
        assert stats.fail_count == 1
        assert abs(stats.pass_rate - 0.5) < 0.001

    def test_benchmark_pass_fail_empty_lists(self):
        """Test error handling for empty inputs."""
        with pytest.raises(ValueError, match="Cannot benchmark empty task list"):
            benchmark_pass_fail([], [])

        with pytest.raises(ValueError, match="Cannot benchmark with empty trials list"):
            benchmark_pass_fail([Mock()], [])

    def test_benchmark_pass_fail_multiple_tasks(self):
        """Test benchmarking multiple tasks."""
        clear_agent_registry()

        # Create two agents with different performance
        good_responses = [
            LLMResponse(thinking="Good at math", code="task_success('4')"),
            LLMResponse(thinking="Good at math", code="task_success('2')"),
        ]
        bad_responses = [
            LLMResponse(thinking="Bad at math", code="task_success('wrong')"),
            LLMResponse(thinking="Bad at math", code="task_success('wrong')"),
        ]

        good_agent = Agent(name="good_agent")
        good_agent.llm_client = DummyLLMClient(responses=good_responses)

        bad_agent = Agent(name="bad_agent")
        bad_agent.llm_client = DummyLLMClient(responses=bad_responses)

        @good_agent.task
        def good_task(question: str) -> str:
            """Good at math."""
            pass

        @bad_agent.task
        def bad_task(question: str) -> str:
            """Bad at math."""
            pass

        trials = [
            Trial(params("What is 2+2?"), expected="4", judge=operator.eq),
            Trial(params("What is 1+1?"), expected="2", judge=operator.eq),
        ]

        results = benchmark_pass_fail([good_task, bad_task], trials)

        assert len(results) == 2
        assert results[good_task].pass_rate == 1.0
        assert results[bad_task].pass_rate == 0.0


class TestBenchmarkNumeric:
    """Test benchmark_numeric function."""

    def test_benchmark_numeric_basic(self):
        """Test basic numeric benchmarking."""
        clear_agent_registry()

        # Responses of different lengths for scoring
        dummy_responses = [
            LLMResponse(
                thinking="Writing", code="task_success('Short story.')"
            ),  # 13 chars = 1.3
            LLMResponse(
                thinking="Writing", code="task_success('Much longer story here!')"
            ),  # 24 chars = 2.4
        ]
        client = DummyLLMClient(responses=dummy_responses)

        agent = Agent(name="writer_agent")
        agent.llm_client = client

        @agent.task
        def write_story(prompt: str) -> str:
            """Write stories."""
            pass

        # Judge based on length
        def length_judge(expected: str, actual: str) -> float:
            return len(actual) / 10.0

        trials = [
            Trial(params("Write short"), expected="story", judge=length_judge),
            Trial(params("Write long"), expected="story", judge=length_judge),
        ]

        results = benchmark_numeric([write_story], trials)

        # Verify results
        assert len(results) == 1
        assert write_story in results

        stats = results[write_story]
        assert isinstance(stats, NumericStats)
        assert stats.total_trials == 2
        assert stats.completed_trials == 2
        assert stats.errored_trials == 0

        # Verify numeric aggregations
        expected_scores = [
            1.2,
            2.3,
        ]  # 'Short story.' = 12 chars, 'Much longer story here!' = 23 chars
        expected_mean = sum(expected_scores) / len(expected_scores)

        assert abs(stats.mean_score - expected_mean) < 0.01
        assert abs(stats.min_score - min(expected_scores)) < 0.01
        assert abs(stats.max_score - max(expected_scores)) < 0.01
        assert abs(stats.total_score - sum(expected_scores)) < 0.01


class TestBenchmarkGeneric:
    """Test benchmark_generic function."""

    def test_benchmark_generic_custom_aggregator(self):
        """Test generic benchmark with custom aggregator."""
        clear_agent_registry()

        # Simple task that returns input
        dummy_responses = [
            LLMResponse(thinking="Echoing", code="task_success(inputs.text)"),
            LLMResponse(thinking="Echoing", code="task_success(inputs.text)"),
        ]
        client = DummyLLMClient(responses=dummy_responses)

        agent = Agent(name="echo_agent")
        agent.llm_client = client

        @agent.task
        def echo_task(text: str) -> str:
            """Echo the input."""
            pass

        # Custom judge that returns dict
        def dict_judge(expected: str, actual: str) -> dict:
            return {
                "exact_match": expected == actual,
                "length": len(actual),
                "uppercase": actual.isupper(),
            }

        # Custom aggregator
        def dict_aggregator(results: list[dict], event_stats: Stats) -> Stats:
            # Just return the base stats for this test
            return event_stats

        trials = [
            Trial(params("hello"), expected="hello", judge=dict_judge),
            Trial(params("WORLD"), expected="WORLD", judge=dict_judge),
        ]

        results = benchmark_generic([echo_task], trials, dict_aggregator)

        stats = results[echo_task]
        assert isinstance(stats, Stats)
        assert stats.total_trials == 2
        assert stats.completed_trials == 2

    def test_benchmark_generic_judge_error(self):
        """Test handling of judge function errors."""
        clear_agent_registry()

        dummy_responses = [LLMResponse(thinking="Test", code="task_success('result')")]
        client = DummyLLMClient(responses=dummy_responses)

        agent = Agent(name="test_agent")
        agent.llm_client = client

        @agent.task
        def test_task(input_val: str) -> str:
            """Test task."""
            pass

        # Judge that always fails
        def failing_judge(expected, actual):
            raise ValueError("Judge failed!")

        def simple_aggregator(results, event_stats):
            return event_stats

        trials = [Trial(params("test"), expected="test", judge=failing_judge)]

        with pytest.raises(TypeError, match="Judge function failed"):
            benchmark_generic([test_task], trials, simple_aggregator)

    def test_benchmark_generic_aggregator_error(self):
        """Test handling of aggregator errors."""
        clear_agent_registry()

        dummy_responses = [LLMResponse(thinking="Test", code="task_success('result')")]
        client = DummyLLMClient(responses=dummy_responses)

        agent = Agent(name="test_agent")
        agent.llm_client = client

        @agent.task
        def test_task(input_val: str) -> str:
            """Test task."""
            pass

        # Aggregator that always fails
        def failing_aggregator(results, event_stats):
            raise ValueError("Aggregator failed!")

        trials = [Trial(params("test"), expected="test", judge=operator.eq)]

        with pytest.raises(ValueError, match="Aggregation failed"):
            benchmark_generic([test_task], trials, failing_aggregator)


class TestConcurrency:
    """Test concurrent execution of benchmarks."""

    def test_concurrent_execution(self):
        """Test that max_concurrency parameter works."""
        clear_agent_registry()

        # This is a basic smoke test - actual concurrency testing would be more complex
        dummy_responses = [
            LLMResponse(thinking="Test 1", code="task_success('1')"),
            LLMResponse(thinking="Test 2", code="task_success('2')"),
            LLMResponse(thinking="Test 3", code="task_success('3')"),
        ]
        client = DummyLLMClient(responses=dummy_responses)

        agent = Agent(name="test_agent")
        agent.llm_client = client

        @agent.task
        def test_task(input_val: str) -> str:
            """Test task."""
            pass

        trials = [
            Trial(params("1"), expected="1", judge=operator.eq),
            Trial(params("2"), expected="2", judge=operator.eq),
            Trial(params("3"), expected="3", judge=operator.eq),
        ]

        # Test with different concurrency levels
        results_seq = benchmark_pass_fail([test_task], trials, max_concurrency=1)
        results_conc = benchmark_pass_fail([test_task], trials, max_concurrency=2)

        # Results should be the same regardless of concurrency
        assert results_seq[test_task].pass_count == results_conc[test_task].pass_count
        assert (
            results_seq[test_task].total_trials == results_conc[test_task].total_trials
        )
