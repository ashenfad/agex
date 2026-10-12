# Examples

Each runs on a real model, named by `AGEX_MODEL` (any model name
pydantic-ai knows, with its key in the environment):

```bash
AGEX_MODEL=anthropic:claude-sonnet-5-5 python examples/session.py
```

| script | what it shows |
|---|---|
| [`session.py`](session.py) | a session: two turns in one workspace, the second remembering the first |
| [`task.py`](task.py) | a task: a typed function answered with a dataclass |
| [`asks.py`](asks.py) | a task that asks its caller a question, and is resumed with the answer |
| [`agent_tasks.py`](agent_tasks.py) | agent-defined tasks: the agent's own script maps a task over a file of reviews |

Each has a `main(agent)`, and `tests/test_examples.py` runs them all on
a scripted model, so they keep working as agex changes.
