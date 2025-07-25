# Examples Overview

The agex examples demonstrate core capabilities and patterns through practical, runnable code. Each example focuses on a specific aspect of the framework.

## Interactive Examples

**[🚀 Interactive Demo](../demo.md)** - Try agex in your browser with our JupyterLite notebooks.

## Core Capabilities

### Function Generation
**File:** `examples/funcy.py`

Agent generates executable Python functions that can be called directly in your program. Demonstrates runtime interoperability beyond JSON serialization.

```python
@agent.task
def fn_builder(prompt: str) -> Callable:
    """Build a callable function from a text prompt."""
    pass

# Agent returns a real Python function
prime_finder = fn_builder("find the next prime larger than a given number")
print(prime_finder(100))  # 101
```

### Mathematical Computing  
**File:** `examples/mathy.py`

Agent performs calculations using Python's math module and works with numerical data. Shows basic agent-module integration.

```python
@agent.task
def transform(prompt: str, numbers: list[float]) -> list[float]:
    """Transform a list of numbers based on a prompt."""
    pass

# Agent transforms data using math functions
radians = transform("Convert degrees to radians", list(range(360)))
```

### Data Visualization
**File:** `examples/viz.py`

Agent works with complex NumPy arrays and generates Plotly visualizations. Bulk data flows between agents without special handling.

```python
@agent.task
def plot_data(prompt: str, data: list[np.ndarray]) -> Figure:
    """Produce a figure from numpy data given the prompt."""
    pass

# Agent creates interactive plots from raw data
plot = plot_data("Show signal trends over time", signal_arrays)
```

## Database Integration

### Raw SQLite Integration
**File:** `examples/db.py`

Agent works directly with `sqlite3.Connection` and `Cursor` objects - no wrapper classes needed. Demonstrates stateful object management.

```python
# Register live database connection
agent.module(conn, name="db", include=["execute", "commit"])

@agent.task  
def query_db(prompt: str) -> Any:
    """Query the database and return results."""
    pass

# Agent executes SQL and returns structured data
results = query_db("Find the oldest user in the database")
```

## Multi-Agent Patterns

### Hierarchical Orchestration
**File:** `examples/multi.py`

An orchestrator agent delegates to specialist sub-agents for data generation and visualization. Shows clean agent coordination.

```python
# Register sub-agents as functions
orchestrator.fn(make_data)
orchestrator.fn(plot_data)

@orchestrator.task
def idea_to_plot(idea: str) -> Figure:
    """Create a plot from a high-level idea."""
    pass

# Orchestrator delegates to specialists automatically
plot = idea_to_plot("Show seasonal sales trends over 10 years")
```

### Iterative Improvement
**File:** `examples/evaluator_optimizer.py`

One agent creates content, another critiques it until quality criteria are met. Demonstrates peer collaboration patterns.

```python
# Simple Python control flow for agent coordination
content = create_content("python decorators")
while (review := review_content(content)).quality != "good":
    content = improve_content(content, review.feedback)
```

## Advanced Patterns

### Agents Creating Agents
**File:** `examples/dogfood.py`

Agents can design and spawn other agents at runtime using the regular agex API. Shows the framework's self-hosting capabilities.

```python
# Register Agent class with architect
architect.cls(Agent, visibility="medium")

@architect.task
def create_specialist(prompt: str) -> Callable:
    """Create an agent task function given a prompt."""
    pass

# Architect creates a math-solving agent
math_solver = create_specialist("solve mathematical equations step by step")
result = math_solver("4x + 5 = 13")
```

## Running the Examples

All examples are self-contained and can be run directly:

```bash
# Clone the repository
git clone https://github.com/USERNAME/agex.git  # TODO: Update when public
cd agex

# Install with your preferred provider
pip install -e ".[openai]"  # or anthropic, gemini

# Run any example
python examples/funcy.py
python -m examples.mathy
```

## GAIA Evaluation Examples

The `examples/gaia/` directory contains examples tested against GAIA benchmark tasks:

- **Logic puzzles** - Complex reasoning and mathematical optimization
- **Multi-step processing** - Data analysis workflows
- **Excel analysis** - Working with spreadsheet data

These demonstrate agex's capabilities on challenging real-world reasoning tasks.

---

**Next:** Try the [Interactive Demo](../demo.md) or dive into the [API Reference](../api/overview.md). 