# Examples Overview

The best way to understand `agex` is to see it in action. These examples showcase the core philosophies of `agex`—from direct library integration to complex multi-agent workflows—and provide a starting point for building your own agents.

All examples are tested with `gpt-4.1-nano` to demonstrate that `agex` works effectively even with smaller, faster models.

---

## 🧩 Core Concepts

### Composition over Tools: `mathy.py`

Most agentic frameworks require you to bundle low-level operations into high-level "tools." If a user asks for something slightly different, you need to write a new tool.

`agex` takes a different approach: you provide the agent with fundamental building blocks, and the agent writes the code to combine them.

The **[`mathy.py`](https://github.com/ashenfad/agex/blob/main/examples/mathy.py)** example shows this. We don't give the agent a `calculate_sine` tool and a `calculate_cosine` tool. We just give it the `math` module. The agent can then handle complex requests in a single pass by composing the primitives.

```python
# Give the agent access to the building blocks
agent.module(math)

@agent.task
def run_calculation(problem: str) -> float:
    """Solve the mathematical problem and return the numeric result."""
    pass

# The agent can now compose functions to solve the problem
# e.g. "what is the sin of pi plus the cosine of 0?"
run_calculation("what is the sin of pi plus the cosine of 0?")
```
This shifts the work of writing composite operations from you to the agent.

### Dynamic Function Generation: `funcy.py`

Agents can generate and return executable Python functions and classes at runtime. In **[`funcy.py`](https://github.com/ashenfad/agex/blob/main/examples/funcy.py)**, an agent is tasked with building a `Callable` from a text prompt. The result is a real Python function you can immediately use in your existing code.

```python
# Agent creates actual Python functions you can use
fn = fn_builder("a fn for the first prime larger than a given number.")

# The returned object is a real callable
my_list = [100, 200, 300]
my_list.sort(key=fn)
```

### Live Object Integration: `db.py`

Agents can work directly with complex, stateful APIs without requiring wrapper classes. `agex` exposes live Python objects—including un-pickleable ones like database connections—while maintaining state serialization safety.

**[`db.py`](https://github.com/ashenfad/agex/blob/main/examples/db.py)** showcases this with raw SQLite integration. The agent works directly with `sqlite3.Connection` and `Cursor` objects. No `DatabaseManager` wrapper is needed—the agent adapts to the existing API.

```python
# Connect to a database and share the instance methods with the agent
conn = sqlite3.connect(...)
agent.module(conn, name="db", include=["execute", "commit"])
```

---

## 🤖 Multi-Agent Patterns

`agex` supports complex systems of specialized agents. Orchestration is done with simple Python control flow—no YAML or complex DSLs required.

### Hierarchical Delegation: `hierarchical.py`

The **[`hierarchical.py`](https://github.com/ashenfad/agex/blob/main/examples/hierarchical.py)** example builds a system where an `orchestrator` agent delegates data generation and plotting tasks to specialist sub-agents to solve high-level visualization ideas. One agent's `task` is exposed as a simple function for another agent to call.

### Peer Collaboration: `evaluator_optimizer.py`

The **[`evaluator_optimizer.py`](https://github.com/ashenfad/agex/blob/main/examples/evaluator_optimizer.py)** example shows two agents collaborating as peers. One agent creates content, and another critiques it. This iterative improvement loop is orchestrated with a simple Python `while` loop.

```python
# Iterative improvement between agents
report = researcher("AI trends in 2024")
while not (review := critic(report)).approved:
    report = writer(review.feedback, report)
```

---

## 🔬 Advanced Patterns

### Agents Architecting Agents: `dogfood.py`

As the ultimate test of library-friendliness, agents can use the `agex` API itself to design and spawn other agents at runtime. This powerful "dogfooding" capability is showcased in **[`dogfood.py`](https://github.com/ashenfad/agex/blob/main/examples/dogfood.py)**.

```python
# Give an architect agent access to the Agent class
architect.cls(Agent)

# The architect can now create new, specialized agents
math_solver = create_specialist("solve mathematical equations step by step")
```

### Empirical Agent Development: `agex.bench`

`agex` includes a built-in benchmarking framework (`agex.bench`) for data-driven agent improvement. You can A/B test different primers, regression-test agent behavior, and measure performance with judge functions that can themselves be agents. This enables systematic agent optimization rather than guesswork.

See **[`benchmarks/funcy_bench.py`](https://github.com/ashenfad/agex/blob/main/benchmarks/funcy_bench.py)** for an example of how to test the `funcy.py` agent.

```python
from agex.bench import Trial, benchmark_pass_fail, params
import operator

# Test agent performance empirically
results = benchmark_pass_fail(
    tasks=[my_agent.solve_problem],
    trials=[
        Trial(params("Calculate 2+2"), expected=4, judge=operator.eq),
        Trial(params("What is 10*5?"), expected=50, judge=operator.eq),
    ],
)
```

---

## 📓 Interactive Notebooks

The **[`agex101.ipynb`](https://github.com/ashenfad/agex/blob/main/docs/demos/agex101.ipynb)** notebook provides a complete, step-by-step walkthrough of a data science task, showing how to reshape raw data, pass `DataFrame` objects, inspect agent "thinking," and create a final visualization.
