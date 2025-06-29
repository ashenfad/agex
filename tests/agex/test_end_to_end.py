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
from agex.llm.core import LLMResponse


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
        LLMResponse(
            thinking="I need to solve this math problem by adding the numbers and multiplying by 2.",
            code="sum_result = add(inputs.x, inputs.y)\nfinal_result = multiply(sum_result, 2)\nexit_success(final_result)",
        )
    ]

    # Use dummy client for predictable responses
    agent.llm_client = DummyLLMClient(responses=responses)

    # Define task
    @agent.task("Solve a math problem involving two numbers.")
    def solve_math_problem(problem: str, x: float, y: float) -> float:  # type: ignore[return-value]
        """
        Solve a mathematical problem using the provided numbers.

        Args:
            problem: Description of the math operation to perform
            x: First number for the calculation
            y: Second number for the calculation

        Returns:
            The numerical result of the mathematical operation
        """
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
        LLMResponse(
            thinking="",
            code="# This response has no thinking section - should trigger parse error\nresult = get_answer()",
        ),
        LLMResponse(
            thinking="I'll call the function to get the answer.",
            code="result = get_answer()\nexit_success(result)",
        ),
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task("Get the answer to the ultimate question.")
    def get_the_answer() -> int:  # type: ignore[return-value]
        """
        Get the answer to the ultimate question of life, the universe, and everything.

        Returns:
            The ultimate answer as an integer
        """
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
        LLMResponse(
            thinking="I'll try to compute something.",
            code="# This will cause a syntax error\nresult = 1 + ",
        ),
        LLMResponse(
            thinking="Let me fix that syntax error.",
            code="result = 1 + 1\nexit_success(result)",
        ),
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task("Compute a simple result.")
    def compute_simple() -> int:  # type: ignore[return-value]
        """
        Perform a simple computation and return the result.

        Returns:
            The computed integer result
        """
        pass

    result = compute_simple()
    assert result == 2


def test_task_with_inputs_access():
    """Test that tasks can access their input parameters."""
    agent = Agent(max_iterations=2)

    responses = [
        LLMResponse(
            thinking="I need to process the input data.",
            code="message = inputs.text.upper()\ncount = inputs.repeat_count\nresult = message * count\nexit_success(result)",
        )
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task("Process text input by transforming and repeating it.")
    def process_text(text: str, repeat_count: int) -> str:  # type: ignore[return-value]
        """
        Transform and repeat input text.

        Args:
            text: The text to transform and repeat
            repeat_count: Number of times to repeat the transformed text

        Returns:
            The processed text result
        """
        pass

    result = process_text(text="hello", repeat_count=3)
    assert result == "HELLOHELLOHELLO"


def test_task_timeout_after_max_iterations():
    """Test that tasks timeout if they exceed max iterations."""
    agent = Agent(max_iterations=2)

    # Response that never calls exit_success
    responses = [
        LLMResponse(
            thinking="I'll do some work but not finish.",
            code='x = 1 + 1\nprint(f"Current value: {x}")',
        )
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task("A task that never completes.")
    def never_ending_task() -> int:  # type: ignore[return-value]
        """
        Execute a task that is designed to timeout (for testing purposes).

        Returns:
            This function should never return normally, but would return an int if it did
        """
        pass

    # Should timeout after max_iterations
    with pytest.raises(TimeoutError, match="exceeded maximum iterations"):
        never_ending_task()


def test_task_with_exit_fail():
    """Test that ExitFail exceptions are properly propagated."""
    agent = Agent(max_iterations=2)

    responses = [
        LLMResponse(
            thinking="I cannot complete this task.",
            code='exit_fail("Task is impossible to complete")',
        )
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task("A task that always fails.")
    def impossible_task() -> str:  # type: ignore[return-value]
        """
        Execute a task that is designed to always fail (for testing error handling).

        Returns:
            This function should never return normally, but would return a string if it did

        Raises:
            ExitFail: Always raised to indicate task failure
        """
        pass

    # Should raise ExitFail
    with pytest.raises(ExitFail) as exc_info:
        impossible_task()

    assert exc_info.value.reason == "Task is impossible to complete"


def test_task_with_no_inputs():
    """Test tasks that don't require any input parameters."""
    agent = Agent(max_iterations=2)

    responses = [
        LLMResponse(
            thinking="This is a simple task with no inputs.",
            code='result = "Hello, World!"\nexit_success(result)',
        )
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task("Return a greeting.")
    def hello_world() -> str:  # type: ignore[return-value]
        """
        Generate a simple greeting message.

        Returns:
            A greeting string
        """
        pass

    result = hello_world()
    assert result == "Hello, World!"


def test_task_with_complex_return_type():
    """Test tasks that return complex data structures."""
    agent = Agent(max_iterations=2)

    responses = [
        LLMResponse(
            thinking="I'll create a dictionary with the requested data.",
            code='result = {"name": inputs.name, "age": inputs.age, "status": "processed"}\nexit_success(result)',
        )
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task("Create a profile dictionary.")
    def create_profile(name: str, age: int) -> dict:  # type: ignore[return-value]
        """
        Create a user profile as a dictionary.

        Args:
            name: The user's name
            age: The user's age

        Returns:
            A dictionary containing the profile information
        """
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
        LLMResponse(
            thinking="I'll use the factorial function to compute the result.",
            code="result = calculate_factorial(inputs.number)\nexit_success(result)",
        )
    ]

    agent.llm_client = DummyLLMClient(responses=responses)

    @agent.task("Compute factorial using registered function.")
    def compute_factorial(number: int) -> int:  # type: ignore[return-value]
        """
        Calculate the factorial of a given number.

        Args:
            number: The number to calculate factorial for

        Returns:
            The factorial result as an integer
        """
        pass

    result = compute_factorial(number=5)
    assert result == 120  # 5! = 120
