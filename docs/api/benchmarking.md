# Benchmarking

The `agex.bench` module provides a framework for empirically evaluating agent performance and primer effectiveness. It enables A/B testing, regression detection, and systematic improvement of agent behavior through data-driven insights.

## Core Concepts

### Trial-Based Evaluation

A **Trial** represents a single test case with:

- **Parameters**: Input arguments to the task function
- **Judge Function**: Evaluates the actual result

### Judge Functions

Judge functions take the actual result from a task and return a new result for aggregation:

- **Pass/Fail**: Return `bool` for success rate metrics
- **Numeric**: Return `float` for average scores
- **Custom**: Return any type with matching aggregator

**Note**: Judge functions can be agent task functions themselves, enabling "agent-as-judge" evaluation patterns where one agent evaluates another's output.

### Metrics

All benchmarks automatically collect:

- **Completion rate**: Successful vs errored trials
- **Performance**: Average actions taken and time per trial
- **Judge-specific**: Pass rates, scores, etc.

## Quick Start

### Simple Pass/Fail Benchmark

```python
from agex.bench import Trial, benchmark_pass_fail, params
import operator

# Define test cases
trials = [
    Trial(
        params=params("Calculate 2 + 2"),
        judge=lambda actual: actual == 4,
    ),
    Trial(
        params=params("Calculate 10 * 5"),
        judge=lambda actual: actual == 50,
    ),
]

# Run benchmark
results = benchmark_pass_fail(
    tasks=[my_agent.solve_math],
    trials=trials,
    max_concurrency=5,
)

# View results
for task, stats in results.items():
    print(f"Pass rate: {stats.pass_rate:.2%}")
    print(f"Average time: {stats.time_per_trial:.2f}s")
```

### Numeric Scoring

```python
from agex.bench import Trial, benchmark_numeric, params

def similarity_scorer(expected_text):
    """Custom judge that returns similarity score."""
    def judge(actual_text):
        # Simple word overlap metric
        expected_words = set(expected_text.lower().split())
        actual_words = set(actual_text.lower().split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
        
        overlap = expected_words & actual_words
        return len(overlap) / len(expected_words)
    return judge

trials = [
    Trial(
        params=params("Summarize: The quick brown fox jumps over the lazy dog."),
        judge=similarity_scorer("A fox jumps over a dog"),
    ),
]

results = benchmark_numeric(
    tasks=[summarizer_agent.summarize],
    trials=trials,
)

for task, stats in results.items():
    print(f"Average score: {stats.mean_score:.2f}")
    print(f"Score range: {stats.min_score:.2f} - {stats.max_score:.2f}")
```

## API Reference

### Core Functions

#### `benchmark_pass_fail`

```python
benchmark_pass_fail(
    tasks: list[Callable[..., T]],
    trials: list[Trial[T, bool]],
    max_concurrency: int = 1,
) -> dict[Callable[..., T], PassFailStats]
```

Benchmark for pass/fail evaluation with boolean judge functions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tasks` | `list[Callable]` | Task functions to benchmark |
| `trials` | `list[Trial]` | Test cases with boolean judges |
| `max_concurrency` | `int` | Maximum concurrent executions |

#### `benchmark_numeric`

```python
benchmark_numeric(
    tasks: list[Callable[..., T]],
    trials: list[Trial[T, float]],
    max_concurrency: int = 1,
) -> dict[Callable[..., T], NumericStats]
```

Benchmark for numeric evaluation with score-based judge functions.

#### `benchmark_generic`

```python
benchmark_generic(
    tasks: list[Callable[..., T]],
    trials: list[Trial[T, U]],
    agg: Callable[[list[U], Stats], Stats],
    max_concurrency: int = 1,
) -> dict[Callable, Stats]
```

Generic benchmark with custom aggregation logic.

### Data Types

#### `Trial[T, U]`

```python
@dataclass
class Trial[T, U]:
    params: Params              # Input parameters
    judge: Callable[[T], U]  # Judge function
```

#### `Params`

```python
@dataclass  
class Params:
    args: tuple[Any, ...]      # Positional arguments
    kwargs: dict[str, Any]     # Keyword arguments

# Convenience constructor
def params(*args, **kwargs) -> Params
```

#### `PassFailStats`

```python
@dataclass
class PassFailStats(Stats):
    pass_count: int           # Successful trials
    fail_count: int          # Failed trials
    
    @property
    def pass_rate(self) -> float  # Success percentage
```

#### `NumericStats`

```python
@dataclass
class NumericStats(Stats):
    mean_score: float        # Average score
    min_score: float         # Minimum score
    max_score: float         # Maximum score
    total_score: float       # Sum of all scores
```

#### `Stats` (Base Class)

```python
@dataclass
class Stats:
    total_trials: int         # Total test cases
    completed_trials: int     # Successfully completed
    errored_trials: int       # Failed with exceptions
    actions_per_trial: float  # Average LLM calls per trial
    time_per_trial: float     # Average execution time per trial
```

## Advanced Usage

### Custom Aggregators

```python
from agex.bench import benchmark_generic, Stats

def custom_aggregator(results: list[str], event_stats: Stats) -> CustomStats:
    """Custom aggregation for string results."""
    return CustomStats(
        **event_stats.__dict__,
        word_count=sum(len(r.split()) for r in results),
        avg_length=sum(len(r) for r in results) / len(results),
    )

results = benchmark_generic(
    tasks=[text_agent.generate],
    trials=trials,
    agg=custom_aggregator,
)
```

### Multi-Agent Comparison

```python
# Compare different agent configurations
tasks = [
    Agent(primer="You are concise.").task(my_task_fn),
    Agent(primer="You are detailed.").task(my_task_fn),
    Agent(primer="You are creative.").task(my_task_fn),
]

results = benchmark_pass_fail(
    tasks=tasks,
    trials=problem_trials,
    max_concurrency=3,
)

# Analyze which primer works best
for task, stats in results.items():
    agent_name = task.__self__.name
    print(f"{agent_name}: {stats.pass_rate:.2%} pass rate")
```

### State and Context Testing

```python
from agex import Versioned

def create_trials_with_state():
    """Generate trials that test stateful interactions."""
    base_state = Versioned({"context": "financial_analysis"})
    
    return [
        Trial(
            params=params("What's the revenue?", state=base_state),
            judge=lambda actual: "revenue_data" in actual,
        ),
        Trial(
            params=params("Calculate the growth rate", state=base_state),
            judge=lambda actual: "growth_calculation" in actual,
        ),
    ]
```

!!! warning "Stateful Benchmarks and Concurrency"
    When designing benchmarks that test stateful interactions (i.e., multiple trials that share the same `Versioned` state object), you **must** use `max_concurrency=1` (the default).

    Using a `max_concurrency` greater than 1 for stateful benchmarks will lead to race conditions and unpredictable results, as concurrent trials will attempt to read from and write to the same state object simultaneously. For stateless trials, concurrency is safe and recommended for performance.

## Example: Complete Benchmark

See `benchmarks/funcy_bench.py` for a complete example that tests function generation capabilities:

```python
"""
Benchmark for examples/funcy.py - Function Generation
Tests agent's ability to generate working Python functions.
"""

def equivalent(expected_fn):
    def judge(actual_fn):
        test_inputs = range(8)
        return all(expected_fn(x) == actual_fn(x) for x in test_inputs)
    return judge

def main():
    trials = [
        Trial(
            params=params("a function that checks if a number is even"),
            judge=equivalent(lambda x: x % 2 == 0),
        ),
        # ... more trials
    ]
    
    results = benchmark_pass_fail(
        tasks=[fn_builder],
        trials=trials,
        max_concurrency=5,
    )
    
    # Print detailed results...

if __name__ == "__main__":
    main()
```

Run benchmarks with:
```bash
python -m benchmarks.funcy_bench
```

## Related Documentation

- [Agent](agent.md) - Agent creation and configuration
- [Task](task.md) - Task function decoration and execution  
- [Events](events.md) - Event system for observability
- [State](state.md) - Persistent state management
