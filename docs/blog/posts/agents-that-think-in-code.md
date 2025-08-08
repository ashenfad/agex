---
title: Agents that think in code (no JSON tools)
date: 2025-08-07
categories:
  - deep-dive
  - examples
author: ashenfad
description: Pass real Python objects, compose libraries in one turn, and keep state. A tour of agex with three short examples.
---

> TL;DR: agex lets agents think in Python code, not JSON tools. They pass real objects (DataFrame, Figure, callables), compose libraries in a single turn, and keep persistent state with full event logs.

![agex demo](../../assets/teaser.gif)

### What you get
- **Real objects** in/out — no brittle JSON adapters
- **Single‑turn composition** — fewer model calls, lower latency
- **Persistent, inspectable state** — checkpoint, time‑travel, debug
- **Secure sandbox** — curated micro‑DSL with AST validation

If you’ve felt the friction of stuffing rich Python workflows into a tool-per-step JSON pipeline, this post is for you.

## A 15‑second taste

```python
import math
from agex import Agent

agent = Agent(primer="You solve math problems.")
agent.module(math, visibility="medium")

@agent.task
def run_calculation(problem: str) -> float:  # type: ignore[return-value]
    """Compute the numeric result."""
    pass

print(run_calculation("sqrt(256) * pi"))  # 50.265482...
```

No tools, no schemas. The agent writes code that calls `math` and returns a real `float`.

## 1) Return real functions (funcy.py)

In many frameworks, creating a function means returning JSON describing code you then compile or reinterpret. With agex, the agent returns an actual Python callable you can use immediately.

What it shows:

- Generate a function from a prompt (e.g., “next prime after n”)
- Use it directly in your code
- Build another function leveraging saved context (`Versioned`)

Read: [`examples/funcy.py`](https://github.com/ashenfad/agex/blob/main/examples/funcy.py)

## 2) Raw SQLite, no wrappers (db.py)

Agents work directly with `sqlite3.Connection` methods. A brief primer teaches the agent to chain `.execute(...).fetch*()` when using persistent state, avoiding unpicklable cursors.

What it shows:

- Register instance methods with `agent.module(conn, name="db", include=[...])`
- Primer‑guided patterns for state constraints
- Natural‑language tasks that create tables, insert rows, and query

Read: [`examples/db.py`](https://github.com/ashenfad/agex/blob/main/examples/db.py), [`examples/db_primer.py`](https://github.com/ashenfad/agex/blob/main/examples/db_primer.py)

## 3) Hierarchical orchestration (hierarchical.py)

The dual‑decorator pattern lets an orchestrator call specialist agents like normal functions. Data flows as real NumPy arrays into a plotting specialist that returns a Plotly `Figure`.

What it shows:

- Orchestrator delegates with simple Python control flow
- Specialists encapsulate capabilities
- Bulk objects pass between agents without serialization glue

Read: [`examples/hierarchical.py`](https://github.com/ashenfad/agex/blob/main/examples/hierarchical.py), [`examples/data.py`](https://github.com/ashenfad/agex/blob/main/examples/data.py), [`examples/viz.py`](https://github.com/ashenfad/agex/blob/main/examples/viz.py)

## Observability you can trust

- `events(state)`: typed events (thinking, code, outputs, errors)
- `Versioned`: automatic checkpoints and diffs; `checkout(commit)` for time‑travel
- `view(agent|state)`: quick inspection of capabilities or memory

These make agent behavior debuggable instead of inscrutable.

## When to reach for agex
- **Python‑native work**: data analysis, plotting, ETL, scientific code
- **Rich objects in/out**: `DataFrame`, `Figure`, arrays, callables
- **Multi‑agent orchestration** without DSLs

Keep tool‑calling for remote services; use agex inside your Python stack when you want power and speed without JSON wrappers.

## Get started

```bash
pip install agex                  # dummy client works out of the box
# or: pip install "agex[openai]"     # choose your provider
```

- **Quick Start**: [ashenfad.github.io/agex/quick-start](https://ashenfad.github.io/agex/quick-start/)
- **Examples**: [ashenfad.github.io/agex/examples/overview](https://ashenfad.github.io/agex/examples/overview/)
- **GitHub**: [github.com/ashenfad/agex](https://github.com/ashenfad/agex)
- **Notebook demo (agex 101)**: [ashenfad.github.io/agex/examples/agex101](https://ashenfad.github.io/agex/examples/agex101/)

If this resonates, share the post — and let me know what you’d like to see next.
