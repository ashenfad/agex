# Task Confirmation: Agent Self-Review of Results

## The Problem

Agents sometimes rush to complete tasks without adequately reviewing their own work. They may call `task_success(result)` prematurely with incomplete, incorrect, or suboptimal results. This leads to:

- **Quality issues**: Solutions that work but aren't robust or well-tested
- **Missed requirements**: Partial fulfillment of complex task specifications  
- **Premature optimization**: Agents stopping at "good enough" rather than pursuing better solutions
- **Lack of self-reflection**: No mechanism to catch their own mistakes before finalizing results

## The Solution: Task-Level Confirmation

A `confirmation=True` parameter on tasks that forces the agent to review and explicitly confirm its own results after calling `task_success()`.

```python
@agent.task(confirmation=True)
def analyze_data(data: DataFrame) -> str:
    """Analyze the dataset and provide insights."""
    # Agent does analysis work...
    analysis_result = "The data shows a 15% increase in sales..."
    task_success(analysis_result)
    # At this point, agent is forced to review its own work
```

### How It Works

1. **Normal Execution**: Agent works on the task as usual
2. **Exit Attempt**: Agent calls `task_success(result)` 
3. **Confirmation Phase**: Instead of immediately exiting, the agent enters a review mode
4. **Self-Review**: Agent is prompted to examine its own result critically
5. **Decision**: Agent can either confirm the result or continue working to improve it

### Example Flow

```python
@agent.task(confirmation=True)
def write_test_cases(function_code: str) -> str:
    """Write comprehensive test cases for the given function."""
    
    # Agent generates initial test cases
    test_code = generate_tests(function_code)
    task_success(test_code)
    
    # Framework intercepts and prompts for confirmation:
    # "Review your test cases. Do they cover all edge cases? 
    #  Are there any missing scenarios? You can continue working 
    #  or confirm this result."
    
    # Agent might respond:
    # "Looking at my tests, I notice I didn't test the empty input case.
    #  Let me add that before finishing."
    
    # Agent continues working...
    improved_tests = add_edge_case_tests(test_code)
    task_success(improved_tests)
    
    # Second review might lead to confirmation
```

## Implementation Considerations

### Confirmation Prompt Design

The framework should provide a structured prompt that encourages thorough self-review:

```
Task Confirmation Required:

You have indicated you want to exit with the following result:
[RESULT PREVIEW]

Before finalizing, please review your work:
- Does this fully address the original task requirements?
- Are there any edge cases or scenarios you haven't considered?
- Could this solution be improved in any meaningful way?
- Have you tested or validated your approach?

You can either:
1. Continue working to improve the solution
2. Confirm this result and exit the task
```
