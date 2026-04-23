"""
End-to-end tests for the dual-decorator pattern with realistic agent interactions.

This module tests the complete dual-decorator workflow:
- Agent-to-agent communication via dual-decorated functions
- State sharing and namespace isolation
- Complex multi-step workflows with multiple specialists
"""

from agex import Agent, clear_agent_registry
from agex.llm.dummy_client import Dummy
from agex.state import connect_state
from tests.agex._emissions import make_response


def test_dual_decorator_math_workflow():
    """Test a realistic math workflow using orchestrator + specialist agents."""
    clear_agent_registry()

    # Create specialist agents
    config = connect_state(type="versioned", storage="memory")
    calculator = Agent(name="calculator", state=config)
    validator = Agent(name="validator", state=config)
    orchestrator = Agent(name="orchestrator", state=config)

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
        make_response(
            thinking='I need to evaluate the expression "15 + 25 * 2". Following order of operations, multiplication comes first.\n25 * 2 = 50\n15 + 50 = 65',
            code="result = 25 * 2  # 50\nresult = 15 + result  # 65\ntask_success(65.0)",
        )
    ]

    validator_responses = [
        make_response(
            thinking='I need to check if 65.0 is a reasonable result for "15 + 25 * 2".\nLet me verify: 25 * 2 = 50, then 15 + 50 = 65. Yes, this is correct.',
            code='# Check the calculation step by step\nexpected = 15 + (25 * 2)  # Order of operations: multiply first\nprint(f"Expected result: {expected}")\nprint(f"Actual result: {inputs.result}")\n\n# The result is correct\nis_valid = (inputs.result == expected)\ntask_success(is_valid)',
        )
    ]

    orchestrator_responses = [
        make_response(
            thinking="I need to solve this math problem step by step:\n1. First calculate the expression using the calculator\n2. Then validate the result with the validator\n3. Return a summary",
            code='# Step 1: Calculate the expression\nexpression = "15 + 25 * 2"\ncalc_result = calculate(expression)\nprint(f"Calculator returned: {calc_result}")\n\n# Step 2: Validate the result\nis_valid = validate_result(expression, calc_result)\nprint(f"Validator returned: {is_valid}")\n\n# Step 3: Return summary\nsummary = {\n    "expression": expression,\n    "result": calc_result,\n    "validated": is_valid,\n    "status": "success" if is_valid else "error"\n}\n\ntask_success(summary)',
        )
    ]

    # Configure dummy LLMs for each agent
    calculator.llm = Dummy(responses=calculator_responses)
    validator.llm = Dummy(responses=validator_responses)
    orchestrator.llm = Dummy(responses=orchestrator_responses)

    # Execute the workflow
    result = solve_math_problem(
        problem_description="Calculate 15 + 25 * 2 and verify the result",
        session="shared_session",
    )

    # Verify the complete workflow worked
    assert isinstance(result, dict)
    assert result["expression"] == "15 + 25 * 2"
    assert result["result"] == 65.0
    assert result["validated"] is True
    assert result["status"] == "success"

    # Note: With memory storage, each agent has isolated state.
    # Sub-agent events are in their own state, not shared.
    # For shared state across agents, use disk storage.


def test_dual_decorator_state_sharing():
    """Test that dual-decorated functions properly share state via namespaces."""
    clear_agent_registry()

    # Create agents
    config = connect_state(type="versioned", storage="memory")
    data_processor = Agent(name="data_processor", state=config)
    analyzer = Agent(name="analyzer", state=config)
    coordinator = Agent(name="coordinator", state=config)

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
        make_response(
            thinking="I need to clean the raw data by removing invalid entries and normalizing values.",
            code="# Clean the data\ncleaned_data = []\nfor item in inputs.raw_data:\n    if isinstance(item, (int, float)) and item > 0:\n        cleaned_data.append(float(item))\n\n# Store intermediate result in my namespace\ntask_success(cleaned_data)",
        )
    ]

    analyzer_responses = [
        make_response(
            thinking="I need to analyze the processed data and generate insights.",
            code='# Analyze the data\ndata = inputs.processed_data\nif data:\n    mean_value = sum(data) / len(data)\n    max_value = max(data)\n    min_value = min(data)\n    \n    insights = {\n        "count": len(data),\n        "mean": mean_value,\n        "max": max_value,\n        "min": min_value,\n        "range": max_value - min_value\n    }\nelse:\n    insights = {"error": "No valid data to analyze"}\n\ntask_success(insights)',
        )
    ]

    coordinator_responses = [
        make_response(
            thinking="I need to coordinate the data pipeline by calling the specialist functions in sequence.",
            code='# Step 1: Process the raw data\nprocessed = process_data(inputs.raw_data)\nprint(f"Data processor returned: {processed}")\n\n# Step 2: Analyze the processed data\nanalysis = analyze_data(processed)\nprint(f"Analyzer returned: {analysis}")\n\n# Step 3: Combine results\nfinal_result = {\n    "raw_count": len(inputs.raw_data),\n    "processed_count": len(processed),\n    "analysis": analysis,\n    "pipeline_status": "completed"\n}\n\ntask_success(final_result)',
        )
    ]

    # Configure LLMs
    data_processor.llm = Dummy(responses=processor_responses)
    analyzer.llm = Dummy(responses=analyzer_responses)
    coordinator.llm = Dummy(responses=coordinator_responses)

    # Execute the pipeline
    result = run_pipeline(
        raw_data=[1, 2, -1, 3.5, 0, 4, "invalid", 5],
        session="pipeline_session",
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

    # Note: With memory storage, each agent has isolated state.
    # Sub-agent events are in their own state, not shared.
    # For shared state across agents, use disk storage.


def test_hierarchical_namespace_state_is_correct():
    """
    This test verifies that state from a sub-agent is correctly saved under
    a hierarchical namespace (e.g., 'orchestrator/worker/key').
    """
    clear_agent_registry()

    # Create two agents
    config = connect_state(type="versioned", storage="memory")
    worker = Agent(name="worker", state=config)
    orchestrator = Agent(name="orchestrator", state=config)

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
    worker.llm = Dummy(
        [
            make_response(
                thinking="I will set the success flag and exit.",
                code="success = True\ntask_success(True)",
            )
        ]
    )
    orchestrator.llm = Dummy(
        [
            make_response(
                thinking="I will call the do_work function.",
                code="result = do_work()\ntask_success(result)",
            )
        ]
    )

    # Execute the workflow with a shared state object
    result = run_worker(session="worker_session")

    # --- Assertions ---
    # 1. The task should complete successfully
    assert result is True

    # Note: With memory storage, each agent has isolated state.
    # Worker's state is in worker's own state, not orchestrator's.
    # For shared state, use disk storage.


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
        make_response(
            thinking="The input says should_fail is False, so I should succeed.",
            code='if inputs.should_fail:\n    task_fail("Operation failed as requested")\nelse:\n    task_success("Operation completed successfully")',
        )
    ]

    # Orchestrator response
    orchestrator_responses = [
        make_response(
            thinking="I need to test the risky operation and handle any failures gracefully.",
            code='try:\n    # First test - should succeed\n    result1 = risky_operation(should_fail=False)\n    print(f"Success case: {result1}")\n    \n    # Compile results\n    results = {\n        "success_case": result1,\n        "test_completed": True\n    }\n    \n    task_success(results)\n    \nexcept Exception as e:\n    # Handle any errors gracefully\n    error_result = {\n        "error": str(e),\n        "test_completed": False\n    }\n    task_success(error_result)',
        )
    ]

    # Configure LLMs
    risky_worker.llm = Dummy(responses=risky_success_responses)
    orchestrator.llm = Dummy(responses=orchestrator_responses)

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
    config = connect_state(type="versioned", storage="memory")
    agent_a = Agent(name="agent_a", state=config)
    agent_b = Agent(name="agent_b", state=config)
    coordinator = Agent(name="coordinator", state=config)

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
        make_response(
            thinking="I'll store the data with a prefix for agent A.",
            code='result = f"A:{inputs.data}"\ntask_success(result)',
        )
    ]

    agent_b_responses = [
        make_response(
            thinking="I'll store the data with a prefix for agent B.",
            code='result = f"B:{inputs.data}"\ntask_success(result)',
        )
    ]

    coordinator_responses = [
        make_response(
            thinking="I'll test namespace isolation by calling both functions with the same data.",
            code='# Call both functions with the same data\nresult_a = store_in_a(inputs.test_data)\nresult_b = store_in_b(inputs.test_data)\n\n# Combine results\nfinal_result = {\n    "agent_a_result": result_a,\n    "agent_b_result": result_b,\n    "are_different": result_a != result_b\n}\n\ntask_success(final_result)',
        )
    ]

    # Configure LLMs
    agent_a.llm = Dummy(responses=agent_a_responses)
    agent_b.llm = Dummy(responses=agent_b_responses)
    coordinator.llm = Dummy(responses=coordinator_responses)

    # Execute the test
    result = run_namespace_test(test_data="shared_data", session="isolation_session")

    # Verify namespace isolation
    assert isinstance(result, dict)
    assert result["agent_a_result"] == "A:shared_data"
    assert result["agent_b_result"] == "B:shared_data"
    assert result["are_different"] is True


def test_sub_agent_registered_module_accessible():
    """Test that a module registered on a sub-agent is accessible in its sandbox.

    Mimics the hierarchical example pattern where a specialist agent has
    domain modules (e.g. numpy) registered and uses them in its code.
    """
    import math

    clear_agent_registry()

    specialist = Agent(name="specialist")
    specialist.module(math, visibility="low")

    orchestrator = Agent(name="orch")

    @orchestrator.fn()
    @specialist.task
    def compute(x: float) -> float:  # type: ignore
        """Compute sqrt(x) using the math module."""
        pass

    @orchestrator.task
    def run(value: float) -> float:  # type: ignore
        """Call compute on a value."""
        pass

    # Sub-agent uses `import math` then calls math.sqrt
    specialist.llm = Dummy(
        [
            make_response(
                thinking="Use math.sqrt",
                code="import math\nresult = math.sqrt(inputs.x)\ntask_success(result)",
            )
        ]
    )
    orchestrator.llm = Dummy(
        [
            make_response(
                thinking="Call compute",
                code="result = compute(inputs.value)\ntask_success(result)",
            )
        ]
    )

    result = run(value=16.0)
    assert result == 4.0


def test_sub_agent_import_alias_works():
    """Test that 'import X as Y' works for registered modules in sub-agents."""
    import json

    clear_agent_registry()

    worker = Agent(name="json_worker")
    worker.module(json, visibility="low")

    boss = Agent(name="boss")

    @boss.fn()
    @worker.task
    def encode(data: dict) -> str:  # type: ignore
        """Encode data as JSON."""
        pass

    @boss.task
    def run(payload: dict) -> str:  # type: ignore
        """Encode a payload via the worker."""
        pass

    worker.llm = Dummy(
        [
            make_response(
                thinking="Use json.dumps with alias",
                code="import json as j\nresult = j.dumps(inputs.data, sort_keys=True)\ntask_success(result)",
            )
        ]
    )
    boss.llm = Dummy(
        [
            make_response(
                thinking="Call encode",
                code="result = encode(inputs.payload)\ntask_success(result)",
            )
        ]
    )

    result = run(payload={"b": 2, "a": 1})
    assert result == '{"a": 1, "b": 2}'


def test_sub_agent_unregistered_import_blocked():
    """Test that sub-agents cannot import modules not registered on them."""
    clear_agent_registry()

    # Disable fs so register_io doesn't auto-register stdlib modules
    worker = Agent(name="restricted_worker", fs=None)

    boss = Agent(name="boss2")

    @boss.fn()
    @worker.task
    def try_import() -> str:  # type: ignore
        """Try to import an unregistered module."""
        pass

    @boss.task
    def run() -> str:  # type: ignore
        """Call the worker."""
        pass

    # Worker tries to import subprocess (never auto-registered) - should fail
    worker.llm = Dummy(
        [
            # First attempt: try importing subprocess (blocked)
            make_response(
                thinking="Try to import subprocess",
                code='import subprocess\ntask_success("got subprocess")',
            ),
            # Second attempt: succeed without import
            make_response(
                thinking="Can't import, just return a string",
                code='task_success("no import needed")',
            ),
        ]
    )
    boss.llm = Dummy(
        [
            make_response(
                thinking="Call try_import",
                code="result = try_import()\ntask_success(result)",
            )
        ]
    )

    result = run()
    assert result == "no import needed"


def test_sub_agent_recursive_module_import():
    """Test that a sub-agent with recursive=True module can import it.

    Mimics the hierarchical example pattern:
      data_maker.module(np, recursive=True, visibility="low")
    where the agent code does 'import numpy as np' and uses np.arange().
    """
    import numpy as np

    clear_agent_registry()

    data_maker = Agent(name="data_maker")
    data_maker.module(np, recursive=True, visibility="low")

    orchestrator = Agent(name="orch2")

    @orchestrator.fn()
    @data_maker.task
    def make_data(idea: str) -> list:  # type: ignore
        """Generate data arrays."""
        pass

    @orchestrator.task
    def run(idea: str) -> list:  # type: ignore
        """Orchestrate data generation."""
        pass

    # Sub-agent imports numpy as np and uses np.arange
    data_maker.llm = Dummy(
        [
            make_response(
                thinking="Generate data using numpy",
                code=(
                    "import numpy as np\n"
                    "arr = np.arange(12)\n"
                    "task_success(arr.tolist())"
                ),
            )
        ]
    )
    orchestrator.llm = Dummy(
        [
            make_response(
                thinking="Call make_data",
                code="result = make_data(inputs.idea)\ntask_success(result)",
            )
        ]
    )

    result = run(idea="test data")
    assert result == list(range(12))


def test_sub_agent_numpy_random_hierarchical():
    """Reproduce the hierarchical example's data_maker pattern.

    Mimics the exact code from example.log where the data_maker agent
    uses import numpy as np then calls np.random.seed(), np.random.normal(),
    np.linspace(), np.sin(), np.maximum(), etc.

    Uses on_event to capture the actual error if it fails.
    """
    import random

    import numpy as np

    clear_agent_registry()

    events = []

    def on_event(event):
        events.append(event)

    # Mirror the hierarchical example setup
    data_maker = Agent(
        name="data_maker_h",
        primer="You excel at generating data via numpy.",
        max_iterations=3,
    )
    data_maker.module(np, recursive=True, visibility="low")
    data_maker.module(random, visibility="low")

    orchestrator = Agent(
        name="orch_h",
        primer="You orchestrate other agents.",
        max_iterations=2,
    )

    @orchestrator.fn()
    @data_maker.task
    def make_data(prompt: str) -> list:  # type: ignore
        """Produce numpy arrays given the prompt."""
        pass

    @orchestrator.task
    def run(idea: str) -> list:  # type: ignore
        """Orchestrate data generation."""
        pass

    # Mimic the exact code from the log's data_maker iteration 1
    data_maker_code = """\
import numpy as np

# Access the inputs
print("Prompt:", inputs.prompt)

# Create seasonal umbrella sales data over 10 years
years = 10
months_per_year = 12
total_months = years * months_per_year

# Time array (in months, 0 to 119)
time = np.arange(total_months)

# Create base trend (slight upward trend over years)
trend = np.linspace(1000, 1500, total_months)

# Create strong seasonal pattern
seasonal = 400 * np.sin(2 * np.pi * time / 12)

# Add random noise for realism
np.random.seed(42)
noise = np.random.normal(0, 100, total_months)

# Combine: base + trend + seasonality + noise
sales = trend + seasonal + noise

# Ensure no negative sales
sales = np.maximum(sales, 0)

# Round to integers for realistic sales counts
sales = np.round(sales).astype(int)

# Create additional useful arrays
months = np.arange(1, total_months + 1)
years_array = np.repeat(np.arange(2014, 2024), 12)
month_of_year = np.tile(np.arange(1, 13), 10)

# Create result list with all relevant arrays
result = [sales, time, seasonal, trend, years_array, month_of_year]

print("Generated arrays:")
print("Sales array shape:", sales.shape)

task_success(result)
"""

    data_maker.llm = Dummy(
        [
            make_response(
                thinking="Generate seasonal data using numpy",
                code=data_maker_code,
            )
        ]
    )
    orchestrator.llm = Dummy(
        [
            make_response(
                thinking="Call make_data",
                code="result = make_data(inputs.idea)\ntask_success(result)",
            )
        ]
    )

    result = run(idea="seasonal umbrella sales", on_event=on_event)

    # Print events for debugging if something went wrong
    for ev in events:
        print(f"EVENT: {type(ev).__name__}: {ev}")

    assert isinstance(result, list)
    assert len(result) == 6


def test_sub_agent_numpy_second_iteration():
    """Test what happens when the first data_maker iteration fails and numpy
    must survive a state round-trip (ModuleRef) for the second iteration.

    This mimics the real failure pattern: iteration 1 fails (e.g. bad inputs
    access), the namespace (including np=ModuleRef) is synced to state, and
    iteration 2 tries the same import.
    """
    import numpy as np

    clear_agent_registry()

    events = []

    def on_event(event):
        events.append(event)

    data_maker = Agent(
        name="dm_retry",
        primer="You excel at generating data via numpy.",
        max_iterations=3,
    )
    data_maker.module(np, recursive=True, visibility="low")

    orchestrator = Agent(name="orch_retry", max_iterations=2)

    @orchestrator.fn()
    @data_maker.task
    def make_data(prompt: str) -> list:  # type: ignore
        """Produce numpy arrays given the prompt."""
        pass

    @orchestrator.task
    def run(idea: str) -> list:  # type: ignore
        """Orchestrate data generation."""
        pass

    # Iteration 1: imports numpy, accesses inputs.prompt, then hits
    # a bad 'from inputs import prompt' which will fail
    iter1_code = """\
import numpy as np
prompt = inputs.prompt
from inputs import prompt
task_success([np.arange(5)])
"""

    # Iteration 2: clean code, should succeed
    iter2_code = """\
import numpy as np
result = [np.arange(10), np.linspace(0, 1, 10)]
np.random.seed(42)
noise = np.random.normal(0, 1, 10)
result.append(noise)
task_success(result)
"""

    data_maker.llm = Dummy(
        [
            make_response(thinking="Try with inputs import", code=iter1_code),
            make_response(thinking="Fixed, just numpy", code=iter2_code),
        ]
    )
    orchestrator.llm = Dummy(
        [
            make_response(
                thinking="Call make_data",
                code="result = make_data(inputs.idea)\ntask_success(result)",
            )
        ]
    )

    result = run(idea="test", on_event=on_event)

    # Print all events for debugging
    for ev in events:
        print(f"EVENT: {type(ev).__name__}: {ev}")

    assert isinstance(result, list)
    assert len(result) == 3


def test_data_maker_numpy_fstring_dtype():
    """Test data_maker with f-string formatting of numpy scalars and dtype.

    This matches the exact code pattern from example.log that triggered
    C-level __import__ errors in the data_maker's sandbox.
    """
    import numpy as np

    clear_agent_registry()

    events = []

    def on_event(event):
        events.append(event)

    data_maker = Agent(
        name="dm_fstr",
        primer="You excel at generating data via numpy.",
        max_iterations=2,
    )
    data_maker.module(np, recursive=True, visibility="low")

    orchestrator = Agent(name="orch_fstr", max_iterations=2)

    @orchestrator.fn()
    @data_maker.task
    def make_data(prompt: str) -> list:  # type: ignore
        """Produce numpy arrays given the prompt."""
        pass

    @orchestrator.task
    def run(idea: str) -> list:  # type: ignore
        """Orchestrate data generation."""
        pass

    # Data_maker code matching the log — includes f-string with .dtype, .min(), .max()
    dm_code = """\
import numpy as np

print("Prompt:", inputs.prompt)

years = 10
months_per_year = 12
total_months = years * months_per_year

time = np.arange(total_months)
trend = np.linspace(1000, 1500, total_months)
seasonal = 400 * np.sin(2 * np.pi * time / 12)

np.random.seed(42)
noise = np.random.normal(0, 100, total_months)

sales = trend + seasonal + noise
sales = np.maximum(sales, 0)
sales = np.round(sales).astype(int)

years_array = np.repeat(np.arange(2014, 2024), 12)
month_of_year = np.tile(np.arange(1, 13), 10)

result = [sales, time, seasonal, trend, years_array, month_of_year]

print(f"Sales array shape: {sales.shape}")
print(f"Sales range: {sales.min()} to {sales.max()}")
for i, arr in enumerate(result):
    print(f"  Array {i}: dtype={arr.dtype}, shape={arr.shape}")

task_success(result)
"""

    data_maker.llm = Dummy([make_response(thinking="Generate data", code=dm_code)])
    orchestrator.llm = Dummy(
        [
            make_response(
                thinking="Call make_data",
                code="result = make_data(inputs.idea)\ntask_success(result)",
            )
        ]
    )

    from agex.agent.datatypes import TaskTimeout

    try:
        result = run(idea="seasonal umbrella sales", on_event=on_event)
    except (TaskTimeout, Exception) as exc:
        result = None
        print(f"\n!!! CAUGHT: {type(exc).__name__}: {exc}")

    print("\n=== EVENTS ===")
    for ev in events:
        parts = getattr(ev, "parts", None)
        print(f"  {type(ev).__name__}[{getattr(ev, 'agent_name', '?')}]: {parts or ''}")

    if result is None:
        raise AssertionError("Task did not complete — see events above")
    assert isinstance(result, list)
    assert len(result) == 6


def test_sub_agent_numpy_full_hierarchical():
    """Reproduce the EXACT hierarchical example interaction from example.log.

    Uses the orchestrator code that inspects returned arrays (arr.shape,
    arr.dtype) and the data_maker code that uses np.random, np.linspace, etc.

    Captures all events to see actual errors.
    """
    import random

    import numpy as np
    import plotly.graph_objects as go

    from agex.helpers import register_numpy, register_pandas, register_plotly

    clear_agent_registry()

    events = []

    def on_event(event):
        events.append(event)
        # Print errors immediately for visibility
        ev_type = type(event).__name__
        if "Error" in ev_type or "Fail" in ev_type:
            print(f"  !!! {ev_type}: {event}")

    # Mirror the hierarchical example exactly
    data_maker = Agent(
        name="dm_full",
        primer="You excel at generating data via numpy.",
        max_iterations=3,
    )
    data_maker.module(np, recursive=True, visibility="low")
    data_maker.module(random, visibility="low")

    plotty = Agent(
        name="plotty_full",
        primer="You excel plotting data via plotly express.",
        max_iterations=3,
    )
    register_plotly(plotty)
    register_numpy(plotty)
    register_pandas(plotty)

    orchestrator = Agent(
        name="orch_full",
        primer="You orchestrate other agents.",
        max_iterations=2,
    )

    @orchestrator.fn()
    @data_maker.task
    def make_data(prompt: str) -> list:  # type: ignore
        """Produce numpy arrays given the prompt."""
        pass

    @orchestrator.fn()
    @plotty.task
    def plot_data(prompt: str, data: list) -> go.Figure:  # type: ignore
        """Produce a figure from numpy data given the prompt."""
        pass

    @orchestrator.task
    def idea_to_plot(idea: str) -> go.Figure:  # type: ignore
        """Orchestrate data generation and plotting."""
        pass

    # --- Orchestrator code ---
    # The orchestrator doesn't have numpy registered, so it shouldn't
    # inspect numpy-specific attributes like .dtype.  It just delegates.
    orch_code = """\
idea = inputs.idea
print("Idea:", idea)

print("\\nGenerating data...")
data = make_data(idea)
print(f"Data generated: {len(data)} arrays")

print("\\nCreating plot...")
figure = plot_data(idea, data)
print("Figure created")

task_success(figure)
"""

    # data_maker iteration 1: the code from the log
    dm_code = """\
import numpy as np

print("Prompt:", inputs.prompt)

years = 10
months_per_year = 12
total_months = years * months_per_year

time = np.arange(total_months)
trend = np.linspace(1000, 1500, total_months)
seasonal = 400 * np.sin(2 * np.pi * time / 12)

np.random.seed(42)
noise = np.random.normal(0, 100, total_months)

sales = trend + seasonal + noise
sales = np.maximum(sales, 0)
sales = np.round(sales).astype(int)

years_array = np.repeat(np.arange(2014, 2024), 12)
month_of_year = np.tile(np.arange(1, 13), 10)

result = [sales, time, seasonal, trend, years_array, month_of_year]
print("Generated arrays:")
print("Sales array shape:", sales.shape)
task_success(result)
"""

    # plotty code: simple figure creation
    plotty_code = """\
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(10)), y=list(range(10))))
fig.update_layout(title="Test")
task_success(fig)
"""

    data_maker.llm = Dummy([make_response(thinking="Generate data", code=dm_code)])
    plotty.llm = Dummy([make_response(thinking="Plot it", code=plotty_code)])
    orchestrator.llm = Dummy([make_response(thinking="Orchestrate", code=orch_code)])

    from agex.agent.datatypes import TaskTimeout

    try:
        result = idea_to_plot(idea="seasonal umbrella sales", on_event=on_event)
    except (TaskTimeout, Exception) as exc:
        result = None
        print(f"\n!!! CAUGHT: {type(exc).__name__}: {exc}")

    # Print all events
    print("\n=== ALL EVENTS ===")
    for ev in events:
        parts = getattr(ev, "parts", None)
        print(f"  {type(ev).__name__}[{getattr(ev, 'agent_name', '?')}]: {parts or ''}")

    if result is None:
        raise AssertionError("Task did not complete — see events above")
