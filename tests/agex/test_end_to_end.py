"""
End-to-end integration tests for the agent system.

These tests verify that all components work together correctly:
- Agent creation and configuration
- Function registration
- Task definition with @agent.task
- Task execution with LLM responses
- Response parsing and code evaluation
- Exit handling and result extraction
"""

import pytest

from agex.agent import Agent, ExitFail
from agex.llm import DummyLLMClient


def test_successful_task_completion():
    """Test complete task execution with successful result."""
    # Create agent with registered functions
    agent = Agent(
        primer="You are a helpful math assistant.",
        timeout_seconds=30.0,
        max_iterations=3,
    )

    @agent.fn()
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @agent.fn()
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    # Define response that completes the task successfully
    responses = [
        """# Thinking
I need to solve this math problem by adding the numbers and multiplying by 2.

```python
sum_result = add(inputs.x, inputs.y)
final_result = multiply(sum_result, 2)
exit_success(final_result)
```"""
    ]

    # Use dummy client for predictable responses
    agent.llm_client = DummyLLMClient(responses=responses)

    # Define task
    @agent.task
    def solve_math_problem(problem: str, x: float, y: float) -> float:  # type: ignore
        """Solve a math problem involving two numbers."""
        pass

    # Execute task and verify result
    result = solve_math_problem(
        problem="Add two numbers and multiply by 2", x=3.0, y=7.0
    )

    assert result == 20.0  # (3 + 7) * 2 = 20


def test_task_with_parse_error_recovery():
    """Test that tasks can recover from malformed LLM responses."""
    agent = Agent(max_iterations=3)

    @agent.fn()
    def get_answer() -> int:
        """Return the answer to everything."""
        return 42

    # First response is malformed (missing thinking section)
    # Second response is correct
    responses = [
        """```python
# This response has no thinking section - should trigger parse error
result = get_answer()
```""",
        """# Thinking
I'll call the function to get the answer.

```python
result = get_answer()
exit_success(result)
```""",
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task
    def get_the_answer() -> int:  # type: ignore
        """Get the answer to the ultimate question."""
        pass

    # Should succeed despite first response being malformed
    result = get_the_answer()
    assert result == 42


def test_task_with_evaluation_error_recovery():
    """Test that tasks can recover from evaluation errors."""
    agent = Agent(max_iterations=3)

    # First response has a syntax error
    # Second response is correct
    responses = [
        """# Thinking
I'll try to compute something.

```python
# This will cause a syntax error
result = 1 + 
```""",
        """# Thinking
Let me fix that syntax error.

```python
result = 1 + 1
exit_success(result)
```""",
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task
    def compute_simple() -> int:  # type: ignore
        """Compute a simple result."""
        pass

    result = compute_simple()
    assert result == 2


def test_task_with_inputs_access():
    """Test that tasks can access their input parameters."""
    agent = Agent(max_iterations=2)

    responses = [
        """# Thinking
I need to process the input data.

```python
message = inputs.text.upper()
count = inputs.repeat_count
result = message * count
exit_success(result)
```"""
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task
    def process_text(text: str, repeat_count: int) -> str:  # type: ignore
        """Process text input by transforming and repeating it."""
        pass

    result = process_text(text="hello", repeat_count=3)
    assert result == "HELLOHELLOHELLO"


def test_task_timeout_after_max_iterations():
    """Test that tasks timeout if they exceed max iterations."""
    agent = Agent(max_iterations=2)

    # Response that never calls exit_success
    responses = [
        """# Thinking
I'll do some work but not finish.

```python
x = 1 + 1
print(f"Current value: {x}")
```"""
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task
    def never_ending_task() -> int:  # type: ignore
        """A task that never completes."""
        pass

    # Should timeout after max_iterations
    with pytest.raises(TimeoutError, match="exceeded maximum iterations"):
        never_ending_task()


def test_task_with_exit_fail():
    """Test that ExitFail exceptions are properly propagated."""
    agent = Agent(max_iterations=2)

    responses = [
        """# Thinking
I cannot complete this task.

```python
exit_fail("Task is impossible to complete")
```"""
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task
    def impossible_task() -> str:  # type: ignore
        """A task that always fails."""
        pass

    # Should raise ExitFail
    with pytest.raises(ExitFail) as exc_info:
        impossible_task()

    assert exc_info.value.reason == "Task is impossible to complete"


def test_task_with_no_inputs():
    """Test tasks that don't require any input parameters."""
    agent = Agent(max_iterations=2)

    responses = [
        """# Thinking
This is a simple task with no inputs.

```python
result = "Hello, World!"
exit_success(result)
```"""
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task
    def hello_world() -> str:  # type: ignore
        """Return a greeting."""
        pass

    result = hello_world()
    assert result == "Hello, World!"


def test_task_with_complex_return_type():
    """Test tasks that return complex data structures."""
    agent = Agent(max_iterations=2)

    responses = [
        """# Thinking
I'll create a dictionary with the requested data.

```python
result = {"name": inputs.name, "age": inputs.age, "status": "processed"}
exit_success(result)
```"""
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task
    def create_profile(name: str, age: int) -> dict:  # type: ignore
        """Create a profile dictionary."""
        pass

    result = create_profile(name="Alice", age=25)

    expected = {"name": "Alice", "age": 25, "status": "processed"}
    assert result == expected


def test_agent_function_visibility_in_task():
    """Test that registered functions are available during task execution."""
    agent = Agent(max_iterations=2)

    # Register a helper function
    @agent.fn()
    def calculate_factorial(n: int) -> int:
        """Calculate factorial of n."""
        if n <= 1:
            return 1
        return n * calculate_factorial(n - 1)

    responses = [
        """# Thinking
I'll use the factorial function to compute the result.

```python
result = calculate_factorial(inputs.number)
exit_success(result)
```"""
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task
    def compute_factorial(number: int) -> int:  # type: ignore
        """Compute factorial using registered function."""
        pass

    result = compute_factorial(number=5)
    assert result == 120  # 5! = 120
