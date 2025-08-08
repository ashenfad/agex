# System Token Counting

Estimate token budgets for the system context before a call: primers, framework instructions, and visibility tiers.

## Concept
- Analyze the static context rendered for an agent and compute token usage per section.
- Support multiple tokenizers (e.g., GPT-4 family) with pluggable adapters.

## Example
```python
from agex import Agent
# from agex import system_token_count  # proposed API

agent = Agent(primer="You are concise and precise.")
# agent.module(numpy, visibility="medium")
# agent.fn(my_custom_function, visibility="high")

# breakdown = system_token_count(agent, model="gpt-4")
# print(breakdown)
```

## Benefits
- Tune visibility and primer size proactively.
- Avoid context overflow and surprise costs.
- Compare registration strategies.

## Considerations
- Token estimates are model/tokenizer-dependent; document caveats.
- Integrate with `view(agent, focus="tokens")` for convenience.

Related issue: [Issue #2](https://github.com/ashenfad/agex/issues/2)
