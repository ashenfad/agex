# Quick start

agex runs an agent over a [nontainer](https://github.com/ashenfad/nontainer)
workspace: a versioned filesystem, a shell and sandboxed Python. You
drive it two ways, a **session** (turn by turn, in a workspace you
keep) and a **task** (a typed call, answered with a value).

## Install

```bash
pip install "agex[anthropic]"    # Anthropic
pip install "agex[openai]"       # OpenAI
pip install "agex[openrouter]"   # OpenRouter
pip install "agex[google]"       # Google
pip install "agex[all]"          # all four
```

Models are called through [pydantic-ai](https://ai.pydantic.dev), which
reads the provider's key from its usual environment variable
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
`GOOGLE_API_KEY`).

## An agent

```python
from agex import Agent
from agex.providers import Settings

agent = Agent(
    "anthropic:claude-sonnet-5-5",          # any model name pydantic-ai knows
    primer="You write small, tested Python.",
    settings=Settings(thinking="low"),      # reasoning effort, where offered
)
```

An `Agent` is a recipe: a model, a primer that opens every request, and
the settings it calls the model with. It holds no state, so one agent
can drive any number of sessions and tasks. `model` may also be a
pydantic-ai `Model` (one pointed at a local OpenAI-compatible server,
say) or a provider of your own. `max_steps` bounds the model calls in a
run, and `max_tool_calls` its tool calls.

## A session

A session drives one workspace. Open the workspace from a nontainer
`Store`, in memory for a try-out or on disk to keep it:

```python
from nontainer import Store

store = Store(memory=True)       # or Store("agex-store") to keep it on disk
ws = store.open("demo")
chat = agent.session(ws)

out = chat.say("Write fib.py printing the first ten Fibonacci numbers, and run it.")
out.status    # "completed"
out.text      # the model's last reply
out.head      # the commit the workspace stands at now
```

The agent's tools are the workspace's: `terminal`, `run_python`,
`file_write` and `file_edit`. What it writes stays in the workspace, so the
next turn finds it, along with the conversation so far:

```python
out = chat.say("Make it print twenty.")
print(ws.files.read("fib.py").decode())
```

**How a turn ends.** `say` returns an `Outcome` and never raises for
how the run went:

| `status` | what happened |
|---|---|
| `completed` | the model answered without calling a tool |
| `cancelled` | `chat.cancel(reason)` stopped it |
| `failed` | an error, or `max_steps` model calls (`out.message` says which) |
| `interrupted` | the provider was overloaded or briefly down; `chat.resume()` carries on in place |

A cancelled or failed turn keeps its work, closed with a note saying
why, so the next turn's model knows where it stopped.

**History.** Tool calls that change the workspace commit as they land,
and each turn's end lands one more. The conversation lives on the
workspace's branch with the files, so going back in history takes the
agent's memory with it:

```python
first = chat.say("Write notes.md with three ideas.")
chat.say("Now delete it.")
ws.checkout(first.head)          # the file is back, and the agent
chat.say("What did we just do?") # remembers only the first turn

child = ws.fork("what-if")       # a branch: files and conversation
agent.session(child).say("Try a different approach.")
```

**Streaming.** `stream` hands back a turn's events as they happen,
from a coroutine:

```python
async for event in chat.stream("Run the tests."):
    if event.kind == "TextDelta":
        print(event.text, end="")
    elif event.kind == "ToolStarted":
        print(f"\n> {event.name}")
print(chat.last.status)
```

Every blocking call has an async twin (`asay`, `aresume`, `awake`), for
code already on an event loop.

## A task

A task is a function whose signature is the contract and whose
docstring is the job. The body stays empty:

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

recipe = plan_dinner(["eggs", "spinach", "feta"])
```

The agent runs on a scratch world of its own, with the inputs bound by
name, and its code ends the task with `task.success(value)`. The value
is checked against the return type and comes back as a `Recipe`.
Inputs and values are data: strings, numbers, lists and dicts of them,
dataclasses, enums, `Literal`s, pydantic models, bytes, tables and
arrays.

A task that can't do the job raises `TaskFailed`. `run` returns the
`Outcome` instead, with the value as `out.value`:

```python
out = plan_dinner.run(["eggs"])
out.status    # "success", "failed", "needs_input", "cancelled" or "interrupted"
```

**On a world of yours.** `world=ws` runs the task on a fork of that
workspace's last commit, so the agent can read its files, and your
workspace is never touched. `keep=True` keeps the fork, and `out.ref`
names it.

**A task that asks.** Its code can stop to ask the caller with
`task.needs_input(question)`. The outcome carries the question, and
`resume` carries on in the same world with the answer:

```python
out = book.run("Somewhere quiet for Friday dinner.")
if out.status == "needs_input":
    print(out.message)           # "How many people are coming?"
    out = out.resume("Four.")
```

Called plainly, a task that asks raises `NeedsInput`.

## Agent-defined tasks

A session's agent can define tasks of its own, for the parts of a
script that need judgment rather than code. Grant `agex` to the
workspace:

```python
from agex import agent_tasks
from nontainer import Profile, PythonConfig

profile = Profile(python=PythonConfig(host_objects={"agex": agent_tasks(agent)}))
ws = store.open("reviews", profile=profile)
```

The agent's code can then write

```python
@agex.task
def sentiment(review: str) -> str:
    """positive, negative or mixed"""

labels = sentiment.map(reviews)
```

Each call is answered by a helper agent that knows only the call: the
docstring, the inputs and the types. `.map` runs several at once, in
order. `@agex.task(objects=())` keeps the world's other host objects
from the helper, for input that comes from someone you don't trust.
For tests, `stand_in_tasks()` answers every call at once without a
model.

## Testing without a model

`ScriptedProvider` replies from a script, so a test runs the real loop
without a model:

```python
from agex.providers.scripted import ScriptedProvider
from nontainer.conformance.corpus import calls, says

agent = Agent(ScriptedProvider([
    calls("run_python", code="print(2 + 2)"),
    says("It's 4."),
]))
assert agent.session(ws).say("What's 2 + 2?").text == "It's 4."
```

## Next

- [Examples](../examples/): runnable scripts for a session, a task, a
  task that asks, and agent-defined tasks
- [Design notes](design.md): the shape, and why
