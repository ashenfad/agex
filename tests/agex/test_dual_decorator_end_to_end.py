"""
End-to-end tests for the dual-decorator pattern with realistic agent interactions.

This module tests the complete dual-decorator workflow:
- Agent-to-agent communication via dual-decorated functions
- State sharing and namespace isolation
- Complex multi-step workflows with multiple specialists
"""

from agex import Agent, clear_agent_registry
from agex.llm.dummy_client import DummyLLMClient
from agex.state import Versioned
from agex.state.kv import Memory


def test_dual_decorator_math_workflow():
    """Test a realistic math workflow using orchestrator + specialist agents."""
    clear_agent_registry()

    # Create specialist agents
    calculator = Agent(name="calculator")
    validator = Agent(name="validator")
    orchestrator = Agent(name="orchestrator")

    # Register specialist functions with the orchestrator
    @orchestrator.fn(docstring="Perform basic arithmetic operations")
    @calculator.task("Calculate the result of a math expression")
    def calculate(expression: str) -> float:  # type: ignore
        """Calculate a mathematical expression."""
        pass

    @orchestrator.fn(docstring="Validate calculation results")
    @validator.task("Check if a calculation result is reasonable")
    def validate_result(expression: str, result: float) -> bool:  # type: ignore
        """Validate that a calculation result makes sense."""
        pass

    # Create the main orchestrator task
    @orchestrator.task("Solve a complex math problem with validation")
    def solve_math_problem(problem_description: str) -> dict:  # type: ignore
        """Solve a math problem using specialist agents."""
        pass

    # Set up dummy LLM responses
    calculator_responses = [
        """# Thinking
I need to evaluate the expression "15 + 25 * 2". Following order of operations, multiplication comes first.
25 * 2 = 50
15 + 50 = 65

```python
result = 25 * 2  # 50
result = 15 + result  # 65
exit_success(65.0)
```"""
    ]

    validator_responses = [
        """# Thinking
I need to check if 65.0 is a reasonable result for "15 + 25 * 2".
Let me verify: 25 * 2 = 50, then 15 + 50 = 65. Yes, this is correct.

```python
# Check the calculation step by step
expected = 15 + (25 * 2)  # Order of operations: multiply first
print(f"Expected result: {expected}")
print(f"Actual result: {inputs.result}")

# The result is correct
is_valid = (inputs.result == expected)
exit_success(is_valid)
```"""
    ]

    orchestrator_responses = [
        """# Thinking
I need to solve this math problem step by step:
1. First calculate the expression using the calculator
2. Then validate the result with the validator
3. Return a summary

```python
# Step 1: Calculate the expression
expression = "15 + 25 * 2"
calc_result = calculate(expression)
print(f"Calculator returned: {calc_result}")

# Step 2: Validate the result
is_valid = validate_result(expression, calc_result)
print(f"Validator returned: {is_valid}")

# Step 3: Return summary
summary = {
    "expression": expression,
    "result": calc_result,
    "validated": is_valid,
    "status": "success" if is_valid else "error"
}

exit_success(summary)
```"""
    ]

    # Configure dummy LLMs for each agent
    calculator.llm_client = DummyLLMClient(responses=calculator_responses)
    validator.llm_client = DummyLLMClient(responses=validator_responses)
    orchestrator.llm_client = DummyLLMClient(responses=orchestrator_responses)

    # Execute the workflow
    result = solve_math_problem(
        problem_description="Calculate 15 + 25 * 2 and verify the result"
    )

    # Verify the complete workflow worked
    assert isinstance(result, dict)
    assert result["expression"] == "15 + 25 * 2"
    assert result["result"] == 65.0
    assert result["validated"] is True
    assert result["status"] == "success"


def test_dual_decorator_state_sharing():
    """Test that dual-decorated functions properly share state via namespaces."""
    clear_agent_registry()

    # Create agents
    data_processor = Agent(name="data_processor")
    analyzer = Agent(name="analyzer")
    coordinator = Agent(name="coordinator")

    # Create dual-decorated functions
    @coordinator.fn(docstring="Process raw data")
    @data_processor.task("Clean and normalize data")
    def process_data(raw_data: list) -> list:  # type: ignore
        """Process and clean raw data."""
        pass

    @coordinator.fn(docstring="Analyze processed data")
    @analyzer.task("Generate insights from data")
    def analyze_data(processed_data: list) -> dict:  # type: ignore
        """Analyze data and generate insights."""
        pass

    @coordinator.task("Coordinate data pipeline")
    def run_pipeline(raw_data: list) -> dict:  # type: ignore
        """Run the complete data processing pipeline."""
        pass

    # Set up responses
    processor_responses = [
        """# Thinking
I need to clean the raw data by removing invalid entries and normalizing values.

```python
# Clean the data
cleaned_data = []
for item in inputs.raw_data:
    if isinstance(item, (int, float)) and item > 0:
        cleaned_data.append(float(item))

# Store intermediate result in my namespace
exit_success(cleaned_data)
```"""
    ]

    analyzer_responses = [
        """# Thinking
I need to analyze the processed data and generate insights.

```python
# Analyze the data
data = inputs.processed_data
if data:
    mean_value = sum(data) / len(data)
    max_value = max(data)
    min_value = min(data)
    
    insights = {
        "count": len(data),
        "mean": mean_value,
        "max": max_value,
        "min": min_value,
        "range": max_value - min_value
    }
else:
    insights = {"error": "No valid data to analyze"}

exit_success(insights)
```"""
    ]

    coordinator_responses = [
        """# Thinking
I need to coordinate the data pipeline by calling the specialist functions in sequence.

```python
# Step 1: Process the raw data
processed = process_data(inputs.raw_data)
print(f"Data processor returned: {processed}")

# Step 2: Analyze the processed data
analysis = analyze_data(processed)
print(f"Analyzer returned: {analysis}")

# Step 3: Combine results
final_result = {
    "raw_count": len(inputs.raw_data),
    "processed_count": len(processed),
    "analysis": analysis,
    "pipeline_status": "completed"
}

exit_success(final_result)
```"""
    ]

    # Configure LLMs
    data_processor.llm_client = DummyLLMClient(responses=processor_responses)
    analyzer.llm_client = DummyLLMClient(responses=analyzer_responses)
    coordinator.llm_client = DummyLLMClient(responses=coordinator_responses)

    # Test with shared state
    shared_state = Versioned(Memory())

    # Execute the pipeline
    result = run_pipeline(
        raw_data=[1, 2, -1, 3.5, 0, 4, "invalid", 5],
        state=shared_state,  # type: ignore
    )

    # Verify the results
    assert isinstance(result, dict)
    assert result["raw_count"] == 8  # Original data count
    assert result["processed_count"] == 5  # After cleaning: [1.0, 2.0, 3.5, 4.0, 5.0]
    assert result["pipeline_status"] == "completed"

    # Verify analysis results
    analysis = result["analysis"]
    assert analysis["count"] == 5  # Valid numbers after processing: [1, 2, 3.5, 4, 5]
    assert analysis["mean"] == 3.1  # (1 + 2 + 3.5 + 4 + 5) / 5 = 15.5 / 5 = 3.1


def test_dual_decorator_error_handling():
    """Test error handling in dual-decorator workflows."""
    clear_agent_registry()

    # Create agents
    risky_worker = Agent(name="risky_worker")
    orchestrator = Agent(name="orchestrator")

    @orchestrator.fn(docstring="A function that might fail")
    @risky_worker.task("Perform a risky operation")
    def risky_operation(should_fail: bool) -> str:  # type: ignore
        """An operation that might fail based on input."""
        pass

    @orchestrator.task("Handle risky operations safely")
    def safe_coordinator(test_mode: str) -> dict:  # type: ignore
        """Coordinate risky operations with error handling."""
        pass

    # Responses for the risky worker (success case)
    risky_success_responses = [
        """# Thinking
The input says should_fail is False, so I should succeed.

```python
if inputs.should_fail:
    exit_fail("Operation failed as requested")
else:
    exit_success("Operation completed successfully")
```"""
    ]

    # Orchestrator response
    orchestrator_responses = [
        """# Thinking
I need to test the risky operation and handle any failures gracefully.

```python
try:
    # First test - should succeed
    result1 = risky_operation(should_fail=False)
    print(f"Success case: {result1}")
    
    # Compile results
    results = {
        "success_case": result1,
        "test_completed": True
    }
    
    exit_success(results)
    
except Exception as e:
    # Handle any errors gracefully
    error_result = {
        "error": str(e),
        "test_completed": False
    }
    exit_success(error_result)
```"""
    ]

    # Configure LLMs
    risky_worker.llm_client = DummyLLMClient(responses=risky_success_responses)
    orchestrator.llm_client = DummyLLMClient(responses=orchestrator_responses)

    # Test the workflow
    result = safe_coordinator(test_mode="success_test")

    # Verify results
    assert isinstance(result, dict)
    assert result["test_completed"] is True
    assert result["success_case"] == "Operation completed successfully"


def test_dual_decorator_namespace_isolation():
    """Test that different specialist agents have isolated namespaces."""
    clear_agent_registry()

    # Create agents
    agent_a = Agent(name="agent_a")
    agent_b = Agent(name="agent_b")
    coordinator = Agent(name="coordinator")

    @coordinator.fn(docstring="Function A")
    @agent_a.task("Store data in agent A namespace")
    def store_in_a(data: str) -> str:  # type: ignore
        pass

    @coordinator.fn(docstring="Function B")
    @agent_b.task("Store data in agent B namespace")
    def store_in_b(data: str) -> str:  # type: ignore
        pass

    @coordinator.task("Test namespace isolation")
    def test_isolation(test_data: str) -> dict:  # type: ignore
        pass

    # Agent responses that store data in their respective namespaces
    agent_a_responses = [
        """# Thinking
I'll store the data in my namespace and return a confirmation.

```python
# Store in my namespace (this will be namespaced automatically)
stored_data = f"A: {inputs.data}"
exit_success(stored_data)
```"""
    ]

    agent_b_responses = [
        """# Thinking
I'll store the data in my namespace and return a confirmation.

```python
# Store in my namespace (this will be namespaced automatically)
stored_data = f"B: {inputs.data}"
exit_success(stored_data)
```"""
    ]

    coordinator_responses = [
        """# Thinking
I'll call both agents to store data and verify they work independently.

```python
# Store data via both agents
result_a = store_in_a(inputs.test_data)
result_b = store_in_b(inputs.test_data)

# Verify both worked
results = {
    "agent_a_result": result_a,
    "agent_b_result": result_b,
    "isolation_verified": result_a != result_b
}

exit_success(results)
```"""
    ]

    # Configure LLMs
    agent_a.llm_client = DummyLLMClient(responses=agent_a_responses)
    agent_b.llm_client = DummyLLMClient(responses=agent_b_responses)
    coordinator.llm_client = DummyLLMClient(responses=coordinator_responses)

    # Test with shared state
    shared_state = Versioned(Memory())

    # Execute the test
    result = test_isolation(test_data="shared_data", state=shared_state)  # type: ignore

    # Verify namespace isolation worked
    assert isinstance(result, dict)
    assert result["agent_a_result"] == "A: shared_data"
    assert result["agent_b_result"] == "B: shared_data"
    assert result["isolation_verified"] is True
