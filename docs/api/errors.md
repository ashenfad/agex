# Error Handling

When you call agent tasks, they may raise specific exceptions based on how the agent completes (or fails to complete) the task. This guide explains these exceptions and how to handle them in your code.

**You don't implement these behaviors** - agents handle completion internally. **You only need to catch the resulting exceptions.**

## Agent Task Exceptions

Agent tasks can raise specific exceptions during execution:

```python
from agex import Agent, TaskFail
# Note: TaskSuccess is handled internally and doesn't need to be imported
```

### `TaskFail`

Raised when an agent determines it cannot complete the task.

**Attributes:**
- `message: str` - The error message provided by the agent

**Example:**
```python
from agex import Agent, TaskFail

agent = Agent()

@agent.task("Impossible task for demonstration")
def impossible_task() -> str:  # type: ignore[return-value]
    pass

try:
    result = impossible_task()
except TaskFail as e:
    print(f"Task failed: {e.message}")
    # Handle the failure appropriately
```

### `TimeoutError`

Raised when an agent exceeds its maximum iterations without completing the task.

**When it occurs:**
- Agent reaches `max_iterations` without finishing
- Usually indicates a framework issue or infinite loop in agent logic
- **Not** a normal error condition - suggests something is wrong

**Example:**
```python
agent = Agent(max_iterations=5)

@agent.task("Task that might loop forever")
def problematic_task() -> int:  # type: ignore[return-value]
    pass

try:
    result = problematic_task()
except TimeoutError as e:
    print(f"Agent timed out: {e}")
    # This suggests a problem with the agent's implementation
    # or the task complexity exceeds the iteration limit
```

## Error Propagation

Error behavior depends on who calls the agent task:

### User Callers

When called directly by user code, agent errors become exceptions:

```python
from agex import Agent, TaskFail

agent = Agent()

@agent.task("Risky operation")
def risky_operation(data: str) -> str:  # type: ignore[return-value]
    pass

# Handle all possible outcomes
try:
    result = risky_operation("input")
    print(f"Success: {result}")
except TaskFail as e:
    print(f"Operation failed: {e.message}")
except TimeoutError as e:
    print(f"Agent timed out: {e}")
```

### Parent Agent Callers

In multi-agent workflows, child agent errors appear in the parent's stdout:

```python
# Create specialist agents
data_processor = Agent(name="data_processor")
orchestrator = Agent(name="orchestrator")

@orchestrator.fn(docstring="Process risky data")
@data_processor.task("Process data with error handling")
def process_risky_data(data: str) -> str:  # type: ignore[return-value]
    pass

@orchestrator.task("Coordinate data processing")
def coordinate_processing(data: str) -> str:  # type: ignore[return-value]
    """Main orchestrator that handles errors from sub-agents."""
    pass

# If the data_processor agent encounters an error, the error message
# will appear in the orchestrator's stdout, allowing it to:
# - Retry with different parameters
# - Use alternative processing methods  
# - Escalate the error by failing itself
```

## Background: How Agents Signal Completion

*This section explains what happens internally when agents run. You don't need to call these functions - they're used by agents automatically.*

Agents have three internal functions to signal different outcomes:

- **`task_success(result)`** - Agent completed successfully and returns the result
- **`task_fail(message)`** - Agent cannot complete the task and provides an error message  
- **`task_continue(*observations)`** - Agent wants to continue to the next iteration with optional observations

These internal agent calls become the exceptions you handle in your code (except for `task_success` which returns the result directly).

## Best Practices

### For Your Code

**Handle expected errors:**
```python
# ✅ Good - comprehensive error handling
try:
    result = agent_task(data)
    process_result(result)
except TaskFail as e:
    log_error(f"Task failed: {e.message}")
    handle_failure()
# Note: TaskSuccess is handled internally and returns the result directly
```

**Distinguish timeout from task failure:**
```python
try:
    result = complex_task(data)
except TaskFail:
    # Task logic determined it couldn't complete - this is expected
    handle_task_failure()
except TimeoutError:
    # Framework issue - agent didn't finish within iteration limit
    log_framework_issue()
    # Consider increasing max_iterations or simplifying the task
```

### For Multi-Agent Workflows

**Error recovery in orchestrators:**
```python
@orchestrator.task("Robust data pipeline")
def robust_pipeline(data: str) -> dict:  # type: ignore[return-value]
    """Orchestrator with error recovery."""
    pass

# The orchestrator agent can implement error recovery logic:
# - Try primary processor first
# - If that fails (error in stdout), try backup processor
# - Return successful result or escalate the error
```

## Integration with State

When agents encounter errors during execution, only recent error information is preserved in state:

```python
from agex import Versioned

state = Versioned()
try:
    result = task_with_errors(state=state)
except TaskFail as e:
    # Check final state - old errors won't accumulate
    final_stdout = state.get("agent_name/__stdout__", [])
    # Only contains recent output, not historical errors
```

## Next Steps

- **Agent Creation**: See [Agent](agent.md) for configuring timeouts and iterations
- **Task Definition**: See [Task](task.md) for implementing agent tasks  
- **Multi-Agent Patterns**: See [Registration](registration.md) for dual-decorator workflows
- **State Management**: See [State](state.md) for error persistence across executions 