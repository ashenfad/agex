"""
End-to-end tests for the dual-decorator pattern with realistic agent interactions.

This module tests the complete dual-decorator workflow:
- Agent-to-agent communication via dual-decorated functions
- State sharing and namespace isolation
- Complex multi-step workflows with multiple specialists
"""

from agex import Agent, clear_agent_registry
from agex.llm.core import LLMResponse
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
        LLMResponse(
            thinking='I need to evaluate the expression "15 + 25 * 2". Following order of operations, multiplication comes first.\n25 * 2 = 50\n15 + 50 = 65',
            code="result = 25 * 2  # 50\nresult = 15 + result  # 65\ntask_success(65.0)",
        )
    ]

    validator_responses = [
        LLMResponse(
            thinking='I need to check if 65.0 is a reasonable result for "15 + 25 * 2".\nLet me verify: 25 * 2 = 50, then 15 + 50 = 65. Yes, this is correct.',
            code='# Check the calculation step by step\nexpected = 15 + (25 * 2)  # Order of operations: multiply first\nprint(f"Expected result: {expected}")\nprint(f"Actual result: {inputs.result}")\n\n# The result is correct\nis_valid = (inputs.result == expected)\ntask_success(is_valid)',
        )
    ]

    orchestrator_responses = [
        LLMResponse(
            thinking="I need to solve this math problem step by step:\n1. First calculate the expression using the calculator\n2. Then validate the result with the validator\n3. Return a summary",
            code='# Step 1: Calculate the expression\nexpression = "15 + 25 * 2"\ncalc_result = calculate(expression)\nprint(f"Calculator returned: {calc_result}")\n\n# Step 2: Validate the result\nis_valid = validate_result(expression, calc_result)\nprint(f"Validator returned: {is_valid}")\n\n# Step 3: Return summary\nsummary = {\n    "expression": expression,\n    "result": calc_result,\n    "validated": is_valid,\n    "status": "success" if is_valid else "error"\n}\n\ntask_success(summary)',
        )
    ]

    # Configure dummy LLMs for each agent
    calculator.llm_client = DummyLLMClient(responses=calculator_responses)
    validator.llm_client = DummyLLMClient(responses=validator_responses)
    orchestrator.llm_client = DummyLLMClient(responses=orchestrator_responses)

    # Use a shared state object to inspect sub-agent stdout
    shared_state = Versioned(Memory())

    # Execute the workflow
    result = solve_math_problem(
        problem_description="Calculate 15 + 25 * 2 and verify the result",
        state=shared_state,  # type: ignore
    )

    # Verify the complete workflow worked
    assert isinstance(result, dict)
    assert result["expression"] == "15 + 25 * 2"
    assert result["result"] == 65.0
    assert result["validated"] is True
    assert result["status"] == "success"

    # Verify that the sub-agent's events are properly logged
    from agex.state import events

    validator_events = events(shared_state, "orchestrator", "validator", children=False)
    assert len(validator_events) > 0

    # Verify that completion events are properly created
    from agex.agent.events import SuccessEvent

    success_events = [e for e in validator_events if isinstance(e, SuccessEvent)]
    assert len(success_events) == 1
    assert success_events[0].agent_name == "validator"
    assert success_events[0].result is True


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
        LLMResponse(
            thinking="I need to clean the raw data by removing invalid entries and normalizing values.",
            code="# Clean the data\ncleaned_data = []\nfor item in inputs.raw_data:\n    if isinstance(item, (int, float)) and item > 0:\n        cleaned_data.append(float(item))\n\n# Store intermediate result in my namespace\ntask_success(cleaned_data)",
        )
    ]

    analyzer_responses = [
        LLMResponse(
            thinking="I need to analyze the processed data and generate insights.",
            code='# Analyze the data\ndata = inputs.processed_data\nif data:\n    mean_value = sum(data) / len(data)\n    max_value = max(data)\n    min_value = min(data)\n    \n    insights = {\n        "count": len(data),\n        "mean": mean_value,\n        "max": max_value,\n        "min": min_value,\n        "range": max_value - min_value\n    }\nelse:\n    insights = {"error": "No valid data to analyze"}\n\ntask_success(insights)',
        )
    ]

    coordinator_responses = [
        LLMResponse(
            thinking="I need to coordinate the data pipeline by calling the specialist functions in sequence.",
            code='# Step 1: Process the raw data\nprocessed = process_data(inputs.raw_data)\nprint(f"Data processor returned: {processed}")\n\n# Step 2: Analyze the processed data\nanalysis = analyze_data(processed)\nprint(f"Analyzer returned: {analysis}")\n\n# Step 3: Combine results\nfinal_result = {\n    "raw_count": len(inputs.raw_data),\n    "processed_count": len(processed),\n    "analysis": analysis,\n    "pipeline_status": "completed"\n}\n\ntask_success(final_result)',
        )
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

    # Import required functions
    from agex.agent.events import SuccessEvent
    from agex.state import events

    # Verify that the orchestrator's completion event is in its own event log
    coordinator_events = events(shared_state, "coordinator", children=False)
    assert len(coordinator_events) > 0

    # Check for SuccessEvent instead of OutputEvents (print statements don't execute when task_success is in same block)
    success_events = [e for e in coordinator_events if isinstance(e, SuccessEvent)]
    assert len(success_events) == 1
    assert success_events[0].agent_name == "coordinator"
    assert isinstance(success_events[0].result, dict)

    # Verify that the sub-agents shared state properly through namespaces
    # Check processor state (agent name is "data_processor")
    processor_events = events(
        shared_state, "coordinator", "data_processor", children=False
    )
    assert len(processor_events) > 0

    # Check analyzer state
    analyzer_events = events(shared_state, "coordinator", "analyzer", children=False)
    assert len(analyzer_events) > 0

    # Verify completion events are properly created for both sub-agents
    processor_success = [e for e in processor_events if isinstance(e, SuccessEvent)]
    assert len(processor_success) == 1
    assert processor_success[0].agent_name == "data_processor"
    assert isinstance(processor_success[0].result, list)

    analyzer_success = [e for e in analyzer_events if isinstance(e, SuccessEvent)]
    assert len(analyzer_success) == 1
    assert analyzer_success[0].agent_name == "analyzer"
    assert isinstance(analyzer_success[0].result, dict)


def test_hierarchical_namespace_state_is_correct():
    """
    This test verifies that state from a sub-agent is correctly saved under
    a hierarchical namespace (e.g., 'orchestrator/worker/key').
    """
    clear_agent_registry()

    # Create two agents
    worker = Agent(name="worker")
    orchestrator = Agent(name="orchestrator")

    # A task for the worker, which the orchestrator can call
    @orchestrator.fn()
    @worker.task("Set a variable in state")
    def do_work() -> bool:  # type: ignore
        """Sets success = True"""
        pass

    # The orchestrator's task that executes the worker's task
    @orchestrator.task("Run the worker")
    def run_worker() -> bool:  # type: ignore
        """Calls do_work()"""
        pass

    # Configure LLM responses
    worker.llm_client = DummyLLMClient(
        [
            LLMResponse(
                thinking="I will set the success flag and exit.",
                code="success = True\ntask_success(True)",
            )
        ]
    )
    orchestrator.llm_client = DummyLLMClient(
        [
            LLMResponse(
                thinking="I will call the do_work function.",
                code="result = do_work()\ntask_success(result)",
            )
        ]
    )

    # Execute the workflow with a shared state object
    shared_state = Versioned(Memory())
    result = run_worker(state=shared_state)  # type: ignore

    # --- Assertions ---
    # 1. The task should complete successfully
    assert result is True

    # 2. The worker's state should be under "orchestrator/worker/".
    worker_success_key = "orchestrator/worker/success"
    assert shared_state.get(worker_success_key) is True, (
        f"Key '{worker_success_key}' not found in state or has wrong value."
    )

    # 3. Verify the state was NOT written to the flat namespace.
    assert shared_state.get("worker/success") is None


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

    # Set up responses
    risky_success_responses = [
        LLMResponse(
            thinking="The input says should_fail is False, so I should succeed.",
            code='if inputs.should_fail:\n    task_fail("Operation failed as requested")\nelse:\n    task_success("Operation completed successfully")',
        )
    ]

    # Orchestrator response
    orchestrator_responses = [
        LLMResponse(
            thinking="I need to test the risky operation and handle any failures gracefully.",
            code='try:\n    # First test - should succeed\n    result1 = risky_operation(should_fail=False)\n    print(f"Success case: {result1}")\n    \n    # Compile results\n    results = {\n        "success_case": result1,\n        "test_completed": True\n    }\n    \n    task_success(results)\n    \nexcept Exception as e:\n    # Handle any errors gracefully\n    error_result = {\n        "error": str(e),\n        "test_completed": False\n    }\n    task_success(error_result)',
        )
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

    # Create agents with separate namespaces
    agent_a = Agent(name="agent_a")
    agent_b = Agent(name="agent_b")
    coordinator = Agent(name="coordinator")

    @coordinator.fn(docstring="Function A")
    @agent_a.task("Store data in agent A namespace")
    def store_in_a(data: str) -> str:  # type: ignore
        """Store data in agent A namespace."""
        pass

    @coordinator.fn(docstring="Function B")
    @agent_b.task("Store data in agent B namespace")
    def store_in_b(data: str) -> str:  # type: ignore
        """Store data in agent B namespace."""
        pass

    @coordinator.task("Test namespace isolation")
    def run_namespace_test(test_data: str) -> dict:  # type: ignore
        """Test namespace isolation and state sharing."""
        pass

    # Set up responses
    agent_a_responses = [
        LLMResponse(
            thinking="I'll store the data with a prefix for agent A.",
            code='result = f"A:{inputs.data}"\ntask_success(result)',
        )
    ]

    agent_b_responses = [
        LLMResponse(
            thinking="I'll store the data with a prefix for agent B.",
            code='result = f"B:{inputs.data}"\ntask_success(result)',
        )
    ]

    coordinator_responses = [
        LLMResponse(
            thinking="I'll test namespace isolation by calling both functions with the same data.",
            code='# Call both functions with the same data\nresult_a = store_in_a(inputs.test_data)\nresult_b = store_in_b(inputs.test_data)\n\n# Combine results\nfinal_result = {\n    "agent_a_result": result_a,\n    "agent_b_result": result_b,\n    "are_different": result_a != result_b\n}\n\ntask_success(final_result)',
        )
    ]

    # Configure LLMs
    agent_a.llm_client = DummyLLMClient(responses=agent_a_responses)
    agent_b.llm_client = DummyLLMClient(responses=agent_b_responses)
    coordinator.llm_client = DummyLLMClient(responses=coordinator_responses)

    # Test the workflow
    shared_state = Versioned(Memory())

    # Execute the test
    result = run_namespace_test(test_data="shared_data", state=shared_state)  # type: ignore

    # Verify namespace isolation
    assert isinstance(result, dict)
    assert result["agent_a_result"] == "A:shared_data"
    assert result["agent_b_result"] == "B:shared_data"
    assert result["are_different"] is True
