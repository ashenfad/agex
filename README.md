# agex

**Agents that work in a versioned world.**

agex is an agent loop over [nontainer](https://github.com/ashenfad/nontainer)
workspaces: a filesystem, a shell and sandboxed Python, with history.
It has two front doors:

- **A session** drives a workspace turn by turn. Its conversation is
  stored in the workspace's branch, so checking out a commit rewinds
  the agent's memory with its files, and a fork carries both.
- **A task** is a typed function an agent runs on a world of its own
  and answers with a value of the declared type.

```bash
pip install "agex[anthropic]"   # or [openai], [openrouter], [google], [all]
```

## A session

```python
from agex import Agent
from nontainer import Store

agent = Agent("anthropic:claude-sonnet-5-5")
store = Store(memory=True)
ws = store.open("demo")

chat = agent.session(ws)
out = chat.say("Write fib.py printing the first ten Fibonacci numbers, and run it.")
print(out.status, out.text)          # completed, and the model's reply
print(ws.files.read("fib.py").decode())   # the file it wrote
out = chat.say("Make it print twenty.")   # it remembers the last turn
```

Tool calls that change the workspace commit as they land, and a turn's
end lands one more commit, stamped with how the run ended: `completed`,
`cancelled`, `failed` or `interrupted`. None of them raise.

## A task

```python
from dataclasses import dataclass

@dataclass
class Recipe:
    name: str
    steps: list[str]
    minutes: int

@agent.task
def plan_dinner(ingredients: list[str]) -> Recipe:
    """A dinner recipe using only these ingredients, plus salt and oil."""
    ...

recipe = plan_dinner(["eggs", "spinach", "feta"])   # a Recipe
```

The docstring is the job, and the body holds no code (`...` is for a
type checker). The agent works in a scratch world (or a fork of
`world=`), and its code hands back the value with `task.success(...)`.
The value is checked against the return type, so the caller gets a
`Recipe` or a `TaskFailed`.

## Agent-defined tasks

A session's agent can define tasks of its own, for the judgment work in
a script: classify, extract, summarize. Grant `agex` to its workspace:

```python
from agex import agent_tasks
from nontainer import Profile, PythonConfig

profile = Profile(python=PythonConfig(host_objects={"agex": agent_tasks(agent)}))
ws = store.open("reviews", profile=profile)
```

and the agent's code can write

```python
@agex.task
def sentiment(review: str) -> str:
    """positive, negative or mixed"""

labels = sentiment.map(reviews)   # helpers run them at once
```

## More

- [Quick start](docs/quick-start.md): install, the first session and
  task, reading an outcome
- [Examples](examples/): runnable scripts for each of the above
- [Design notes](docs/design.md): the shape, and why
- [Plan](docs/plan.md): what is done and what is next

agex 0.13 is a new API. The 0.12 line, agents working directly with an
existing Python codebase, is at the
[`v0.12.4`](https://github.com/ashenfad/agex/tree/v0.12.4) tag and on
PyPI as `agex==0.12.4`.

## Development

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).
