# Error Handling

When you call agent tasks, they may raise specific exceptions based on how the agent completes (or fails to complete) the task. This guide explains these exceptions and how to handle them in your code.

**You don't implement these behaviors** - agents handle completion internally. **You only need to catch the resulting exceptions.**

## Agent Task Exceptions

Agent tasks can raise specific exceptions during execution:

```python
from agex import Agent, TaskFail, TaskClarify, TaskTimeout
# Note: TaskSuccess is handled internally and doesn't need to be imported
```

### `TaskClarify`

Raised when an agent needs more information from the caller (a human or another agent) to proceed. This is a non-terminal, interactive signal.

**Attributes:**
- `message: str` - The agent's question or request for clarification

**Example:**
```python
from agex import Agent, TaskClarify

agent = Agent()

@agent.task("Perform an action that might require confirmation.")
def confirmable_action(action: str) -> str:  # type: ignore[return-value]
    pass

prompt = "delete all files"
while True:
    try:
        result = confirmable_action(prompt)
        print(f"Success: {result}")
        break
    except TaskClarify as e:
        # The agent is asking for confirmation.
        response = input(f"{e.message} (y/n)? ").lower()
        if response == 'y':
            # Add the confirmation to the prompt and retry.
            prompt += " -- user confirmed."
            print("Retrying with confirmation...")
        else:
            print("Action cancelled.")
            break
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

### `TaskTimeout`

Raised when an agent exceeds its maximum iterations without completing the task.

Think of this less as a recoverable error and more as a **signal to the developer** that something is wrong. An agent should ideally complete its task or fail gracefully (`TaskFail`) well within the iteration limit. A timeout suggests the agent is stuck in a loop, the task is too complex for the current `max_iterations` setting, or there is an issue in the framework itself.

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
except TaskTimeout as e:
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
except TaskTimeout as e:
    print(f"Agent timed out: {e}")
```

### Parent Agent Callers

In multi-agent workflows, child agent errors are automatically converted to evaluation errors that appear in the parent's execution log (its virtual `stdout`). This allows parent agents to see and respond to sub-agent failures naturally.

**How it works:**
- When a sub-agent calls `task_clarify()` or `task_fail()`, the framework converts these to `EvalError`s
- The parent agent sees these errors in its execution log as: `💥 Evaluation error: Sub-agent needs clarification: <message>` or `💥 Evaluation error: Sub-agent failed: <message>`
- The parent can then respond by retrying with different parameters, using alternative approaches, or escalating the error

This error conversion only happens for sub-agents. Top-level agents (called directly by user code) still raise `TaskClarify` and `TaskFail` exceptions normally.

## Background: How Agents Signal Completion

*This section explains what happens internally when agents run. You don't need to call these functions - they're used by agents automatically.*

Agents have internal functions to signal different outcomes:
- **`task_success(result)`** - Agent completed successfully. Becomes the return value of the task.
- **`task_fail(message)`** - Agent cannot complete the task. Raises a `TaskFail` exception.
- **`task_clarify(message)`** - Agent needs more information. Raises a `TaskClarify` exception.
- **`task_continue(*observations)`** - Agent wants to continue to the next think-act cycle. This is the default internal behavior and does not raise an exception.

These internal agent calls become the exceptions you handle in your code.

## Best Practices

### For Your Code

**Handle expected errors and clarifications:**
```python
# ✅ Good - comprehensive error handling
try:
    result = agent_task(data)
    process_result(result)
except TaskClarify as e:
    handle_clarification(e.message)
except TaskFail as e:
    log_error(f"Task failed: {e.message}")
    handle_failure()
```

**Distinguish timeout from task failure:**
```python
try:
    result = complex_task(data)
except TaskClarify as e:
    # Handle the agent's request for more information
    handle_clarification(e.message)
except TaskFail:
    # Task logic determined it couldn't complete - this is expected
    handle_task_failure()
except TaskTimeout:
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


## Next Steps

- **Agent Creation**: See [Agent](agent.md) for configuring timeouts and iterations
- **Task Definition**: See [Task](task.md) for implementing agent tasks  
- **Multi-Agent Patterns**: See [Registration](registration.md) for dual-decorator workflows
- **State Management**: See [State](state.md) for error persistence across executions
