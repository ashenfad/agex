# System Token Counting: Static Context Analysis

## The Need

Users need visibility into how their agent configuration choices affect token consumption. Currently, there's no way to understand the token cost of:
- Agent primers
- Function registrations at different visibility levels  
- Framework system instructions
- Overall "baseline" token usage before conversation begins

Without this visibility, users can't optimize their agent configurations for cost or context efficiency.

## The Solution: `system_token_count()`

A utility function that analyzes the **static context** - everything that goes into the system message before any conversation begins.

```python
from agex import Agent, system_token_count

agent = Agent(primer="You are a helpful assistant...")
agent.module(numpy, visibility="medium") 
agent.fn(my_custom_function, visibility="high")

breakdown = system_token_count(agent, model="gpt-4")
print(breakdown)
```

### Example Output

```
System Message Token Breakdown (gpt-4 tokenizer):
┌─────────────────────────┬────────┐
│ Component               │ Tokens │
├─────────────────────────┼────────┤
│ Agent primer            │    120 │
│ Framework instructions  │    340 │
│ High visibility defs    │  1,200 │
│ Medium visibility defs  │    800 │
│ Low visibility defs     │      0 │ (hidden)
├─────────────────────────┼────────┤
│ Total system context    │  2,460 │
│ Remaining budget        │  1,636 │ (4096 max)
└─────────────────────────┴────────┘
```

### Actionable Insights

This enables users to:
- **Optimize visibility levels**: "High visibility costs 1,200 tokens - should I move some to medium?"
- **Plan context budgets**: "I have 1,636 tokens left for conversation and task context"  
- **Compare registration strategies**: "This module registration costs X tokens per interaction"
- **Avoid context overflow**: Warnings when static context consumes too much budget

## Implementation Considerations

### Token Counting
- Support different tokenizers (gpt-4, gpt-3.5-turbo, etc.)
- Use the same tokenization logic as the actual framework
- Account for formatting and separators in system messages

### Breakdown Granularity
Could provide different levels of detail:
```python
# Summary view (default)
system_token_count(agent)

# Detailed breakdown by registration
system_token_count(agent, detailed=True)
# Shows individual functions/modules and their token costs

# Interactive analysis
system_token_count(agent, interactive=True)  
# Shows before/after for registration changes
```

### Integration with `view()`
Natural extension of the existing `view()` functionality:
```python
from agex import view

# Current: shows what agent sees
view(agent)

# Future: shows token breakdown
view(agent, focus="tokens")
```

## Static vs Dynamic Context

This addresses the **controllable** portion of token usage:

**Static Context (this feature):**
- Agent primer: Fixed per configuration
- Framework instructions: Fixed per agex version
- Function definitions: Fixed per registration strategy
- = Optimizable by user right now

**Dynamic Context (future - see faded-memory.md):**
- Conversation history: Grows over time
- Agent actions/stdout: Accumulates per task
- State change summaries: Expands with usage
- = Requires sophisticated memory management

## Value Proposition

Gives users immediate control over a significant portion of their token costs. While LLM inference time dominates overall latency, token efficiency directly impacts:
- Cost per interaction
- Context available for conversation
- Ability to register extensive capabilities

This is a high-value, implementable feature that complements the long-term faded memory work. 