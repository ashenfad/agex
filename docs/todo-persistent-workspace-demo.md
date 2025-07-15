# TODO: Demonstrate the Persistent Agent Workspace

This document captures the plan to highlight `agex`'s unique ability to create evolving agents that build and retain their own tools over time, using the `Versioned` state as a persistent compute environment.

## 1. Feature Narrative: "Evolving Agents with Persistent Compute"

The core idea is to frame this capability not just as memory, but as a form of agent evolution. The agent doesn't just remember what it said; it remembers what it *learned to do*. This is a significant architectural advantage over frameworks that rely on stuffing tool source code into the prompt.

Key advantages to emphasize:
- **Scalability:** The `Versioned` approach consumes minimal prompt tokens, unlike including full function source code in every call, which quickly exhausts the context window.
- **Fidelity & Reliability:** The agent reuses a stable, "live" Python object, not a "photocopy" of code that could be regenerated with subtle errors.
- **Statefulness:** Persisted functions can be closures that hold state (e.g., an internal cache or counter), which is impossible with text-based function "memory".
- **Abstraction:** Allows the agent to build and trust its own abstractions, freeing up cognitive load to focus on higher-level problems.

## 2. Proposed `README.md` Section

This text should be added to the "What Makes This Different" section of the main `README.md`.

> ### **Evolving Agents with a Persistent Compute Environment**
>
> Most frameworks remember chat history. `agex` goes a step further by treating an agent's entire compute environment—including helper functions it writes for itself—as part of its versioned memory. Using a "micro-git" model for its state, an `agex` agent can build and reuse its own tools over time, becoming more efficient and capable with every interaction. This transforms agents from simple tool-users into evolving tool-builders.

## 3. Proposed New Example: `examples/parser_agent.py`

This example is designed to be more compelling than extending an existing one. It demonstrates the value of the agent creating a tool *for itself* to solve a non-trivial, recurring task.

```python
# examples/parser_agent.py
from agex import Agent, Versioned, view

# A primer that encourages defining and reusing helper functions
PRIMER = """
You are an expert at data parsing. When faced with a complex format,
define a helper function to handle the parsing logic. You will be able to
reuse this function in the future to save time and effort.
"""

parser_agent = Agent(primer=PRIMER)

@parser_agent.task
def parse_log_data(log_lines: list[str]) -> list[dict]:
    """Parse the given log lines and return a list of structured dictionaries."""
    pass

def main():
    state = Versioned()
    log_data_part1 = [
        "ID:001|USER:jdoe|ACTION:LOGIN|TS:1678886400",
        "ID:002|USER:sroe|ACTION:LOGOUT|TS:1678886405",
    ]

    print("--- First Run: Learning to Parse ---")
    parsed_data1 = parse_log_data(log_lines=log_data_part1, state=state)
    print(f"Parsed {len(parsed_data1)} records.")
    print("Agent's thought process: 'I need to write a new function for this format.'")
    # In a real run, use view() to show the agent generated a new helper function here.


    print("\n--- Second Run: Reusing the Helper Function ---")
    log_data_part2 = [
        "ID:003|USER:psmith|ACTION:UPDATE_PROFILE|TS:1678886410",
    ]
    parsed_data2 = parse_log_data(log_lines=log_data_part2, state=state)
    print(f"Parsed {len(parsed_data2)} records.")
    print("Agent's thought process: 'I already have a function for this. I will reuse it.'")
    # In a real run, use view() to show the agent generated a much simpler script
    # that just calls the previously defined helper function.

if __name__ == "__main__":
    main()
```

This clearly separates the concept of "runtime interoperability" (agent gives tool to developer) from "agent evolution" (agent builds tool for itself). 