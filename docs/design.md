# agex design

What agex is shaped like, and why. agex 0.13.0 is the first release of
this design (2026-10-12); `docs/plan.md` says what comes next. The
README and `docs/quick-start.md` show how to use it.

## Why

agex and its TypeScript sibling share one idea: **a versioned world
makes quick-loop agent work possible.** One kvgit branch holds the
files, the cache, the working directory and the conversation. One
commit captures all of it, so a rewind is a checkout and a fork is
O(1). The agent acts by writing code and running a shell, and finished
work leaves as a frozen subtree.

There are two stacks, with different fixed constraints:

| | loop | code runs in | world |
|---|---|---|---|
| agex (Python) | agex | sandtrap, or a dud VM | nontainer: a kvgit branch and a VirtualFS |
| agex-ts / agex-studio | agex-ts, in the browser tab | a Web Worker | kvgit-ts and KvgitFS, in IndexedDB |

- **agex-studio stays serverless.** Apps are published straight to
  tablets, and people build their own with nothing but an OpenRouter
  key.
- **Server-side use is Python, on nontainer.** nontainer-studio is
  where features are proven first.

So there are two codebases, kept aligned on **meaning** (what a fork,
a merge, a delegate and an outcome do), not on bytes. Their storage
formats may differ until something has to cross between them, and
nothing does yet.

## Vocabulary

- **World:** state. kvgit plus a file layout plus planes: files, cache,
  working directory, conversation.
- **Kernel:** where code runs: sandtrap in-process or under process
  isolation, a dud VM, or a Web Worker in TS. nontainer calls it an
  `Executor`, and executors never commit.
- **Loop:** what drives the model. agex, or agno through nontainer's
  adapter.
- **Host:** effects that leave the world: a database, the web, the
  network. Everything inside the world is versioned, so it is a value.

World, kernel and host are nontainer's. Only the loop is agex's.

The unit that crosses all four:

```
run(task, world=ref, inputs, policy) → events…, Outcome(ref′, value?, status)
```

Each concept is that `run` with different defaults:

- **a chat turn:** the same branch, the conversation carried, text back;
- **a typed task:** a fresh fork or scratch world, no transcript, a
  value checked against its type;
- **a delegate:** a fork the parent decides what to do with;
- **a question:** an outcome that needs input, continued with `resume`.

## Two front doors

**An agent is a loop, and the world gives it a body.** `Agent` holds a
model, a primer, the request settings and the run's limits
(`max_steps`, `max_tool_calls`, compaction), plus a default profile
used only for worlds it creates itself. It holds no state.

| | session | task |
|---|---|---|
| world | the session's branch, changed in place | a fork of `world=`, or a scratch world; the caller's is untouched |
| conversation | carried across turns | fresh |
| result | the model's text, and the new head | a value of the declared type, and the fork's ref |
| ends when | the model answers without calling a tool | its code calls `task.success(value)` |

So "functional" is exact: a task is pure with respect to the caller's
world, and its only effects go through host objects.

```python
agent = Agent("anthropic:claude-sonnet-5-5", primer="You build study apps.")

chat = agent.session(ws)                     # side effects, in place
out = chat.say("make a flashcard app")       # Outcome: status, text, head, events
async for event in chat.stream("add a timer"): ...
chat.inbox.put(note)                         # steering, mid-turn

@agent.task                                  # a typed call
def grade(responses: list[Response]) -> Report:
    """Grade the quiz."""
    ...

report = grade(rs)                           # on a scratch world
out = grade.run(rs, world=ws, keep=True)     # on a fork of ws: .value .ref .status
```

Each blocking call has an async twin (`asay`, `arun`, …), and a task
may be an `async def`. The loop underneath is async.

## Sessions

A turn is one run inside nontainer's `ws.turn`. The model is asked, the
tools it calls run against the workspace, and the loop goes round until
it answers without calling one. Tool calls that change the workspace
commit as they land, and the turn's end lands one more commit, stamped
with how the run ended.

- **The run record is agex's own** (`agex.record`): plain frozen
  dataclasses with a `kind` on every union member, read and written as
  JSON, stored one body per run in the workspace's conversation plane
  under nontainer's index. Provider message types are mapped at the
  edge and never stored, so the record is also what TS mirrors.
- **What a turn streams is nontainer's `TurnEvent`,** shared by every
  harness and the studio, so a consumer reads agno and agex runs alike.
- **Endings:**
  - `completed`;
  - `cancelled`: `cancel(reason)` stops the turn and interrupts the
    code running in the workspace;
  - `failed`: an error, or `max_steps`;
  - `interrupted`: a provider error worth retrying, which `resume`
    continues in place, and `abandon(reason)` ends.

  A cancelled or failed turn keeps its work, closed with a note saying
  why, so the next turn's model knows where it stopped. None of the
  endings raise.
- **Steering:** a note put in `inbox` mid-turn rides the next tool
  result.
- **A session reads only its own loop's conversations.** A workspace
  agno wrote is refused, because its runs are in agno's format.

## Tasks

### Outcomes through a host object

`task.success(value)`, `task.fail(reason)` and
`task.needs_input(question)` are methods of a `task` host object that
the loop binds into the task's world. nontainer needs no knowledge of
agex: a task's fork gets it through `Store.fork(..., profile=...)`.

- **A value that doesn't fit is a `TypeError` at the call,** so the
  agent fixes it within the same script. Old agex validated only after
  the step ended.
- **The call stops the script,** by raising a `BaseException` that
  agent code's `except Exception` doesn't swallow. Files written before
  it are kept.
- **The loop trusts the recorded value, not the script's error.** On
  dud, a host-side error reaches the guest as a plain `RuntimeError`,
  which code could swallow; the recorded value still wins.
- **A model that stops without finishing** is sent back to finish,
  twice. Then the task fails.

### Calls and worlds

- **Statuses:** `success` with a value, `failed`, `needs_input` with
  the question, `cancelled` and `interrupted`. A plain call returns the
  value, or raises `TaskFailed`, `NeedsInput` or `TaskInterrupted`.
- **Worlds:**
  - by default, a scratch world in a memory store, built from
    `agent.profile` and dropped afterwards;
  - `world=ws` forks that workspace's last commit, with `task` and the
    inputs added to its profile;
  - `keep=True` keeps the fork, and `out.ref` names it.
- **Resuming.** A `needs_input` outcome keeps its world whatever
  `keep=` says. `out.resume(answer)` continues it in this process;
  `task.resume(ref, answer, world=ws)` continues it by ref, after a
  restart too on a store that persists. Its inputs come back from the
  world, except a live one, which is passed again.
- **A typed call.** A call reads as the decorated function's return
  type to a type checker; its arguments are checked when it is made.

### What the model sees

Python, never a schema:
- the signature, and the source of the types it names;
- a preview of each input (a table's columns and first rows, an
  array's shape and dtype);
- for a live input, how to use it: a function by its signature and
  docstring, an object by its class's docstring and public methods.

## Task values

How a task's inputs and its value travel between the caller, the kernel
and the world, on every rung.

**JSON Schema can't be the contract.** It covers only the data subset
of Python types: plain classes, tables and callables have none, and a
union with an opaque class silently loses that branch. **Pickle can't
be the transport** off in-process, because it rebuilds whatever the
bytes ask for.

### The declared type picks the encoding and drives decoding

The decoder only ever builds the type the task declared, never what
the payload claims.

| kind | examples | on the wire | where it can go |
|---|---|---|---|
| data | primitives, containers, dataclasses, TypedDict, pydantic models, enums, `Literal`, datetime | JSON | everywhere, including apps and TS |
| table | DataFrame, Series, `pa.Table`, anything with `__arrow_c_stream__` | Arrow IPC | wherever pyarrow (or apache-arrow in TS) is |
| array | `ndarray` | `.npy`, loaded without pickle | any Python rung |
| bytes and files | `bytes`, a file the agent wrote | bytes or a file ref | everywhere |
| live | callables, clients, generators | none | in-process only |

- **Nested values work.** A value is a JSON tree whose leaves may be
  tagged binary parts, and a tag is accepted only where the declared
  type allows that kind: `dict[str, DataFrame]` comes back as one.
- **`Any`, or no annotation,** means any data off in-process, decoded
  as plain JSON values.
- The encoding is nontainer's (`nontainer.values`), generalizing the
  one app handlers return through. dud's JSON, bytes and file values
  map onto it one to one.

### Two classes of world, checked early

**In-process carries anything. Every other rung carries exactly the
encodable kinds:** process isolation, dud, apps, TS, and anything
stored for a later resume. Process isolation uses the encoding too,
not a restricted unpickler, so "works under process isolation" implies
"works on dud", and moving a world between them never breaks a task.

The check runs as soon as the rung is known, never mid-run: when a task
is called, against the world it will run in, before any model call; and
in the kernel, when agent code defines a task there. The refusal names
the type, the rung and the ways out.

**Validation is by the declared type, the same on every rung.** What
agent code passes to `task.success` is encoded and decoded by the
return type, in-process too, so the caller gets it built afresh as the
declared type: a list passes for a tuple, a dict with a record's fields
for the record, and `"42"` is refused for an `int`.

### Inputs

- **Inputs are values on every rung.** Each input reaches the task's
  code as a fresh copy each run (`nontainer.values.copy` in-process,
  pickled in off it, which is the safe direction), so a task never
  changes what its caller passed.
- **A live input is a capability.** A client or callable passed as an
  argument is bound as a host object of the task's world, under the
  world's host-object policy, and reached through a proxy where the
  code runs elsewhere. It can't be stored, so a resume after a restart
  needs it passed again.

### The `__task__` plane

A reserved plane beside `__conversation__`, which the task's loop
writes, so each write lands in the commit of the run it belongs to:

```
__task__/spec            name, instructions, each type (and its schema), which inputs are stored
__task__/inputs/<name>   an input sent by value, encoded
__task__/state           running, success, failed (with the reason) or needs_input (with the question)
__task__/value           the value, encoded, when it can be stored
```

A resume, or a parent reading a delegate's ref, reads it there. A task
starts its plane afresh, so a world forked from another task's world
doesn't carry that task's plane.

## Agent-defined tasks

Delegation from inside a world comes in two shapes, each with its own
door:
- **Side effects** (work in a world, report back in prose) go through
  the `sessions` tool (see "Delegation").
- **Functions** (a typed value from inputs) go through code. Agent code
  defines a task the way an embedder does, and calls it from a script,
  a helper module or an app's handler.

```python
@agex.task
def corners(shape: str) -> list[Point2D]:
    """The corner points of the named shape, counterclockwise."""

pts = corners("unit square")              # its own Point2D objects, or TaskFailed
many = corners.map(["square", "hexagon"]) # at once, results in order

@agex.task(objects=())                    # the helper holds none of the
def summarize(text: str) -> str:          # world's host objects
    """A one-line summary of the text."""
```

### A host object, not a primitive

`agex` is a stubbed host object an embedder grants
(`agent_tasks(agent)`), under a name of its choosing. Agent code holds
only the stub; the host half holds the embedder's `Agent`, which agent
code never reaches. So code gets no model client, no embedder profile,
and nothing to configure past a primer and `objects=`. Any harness on
nontainer can grant it, agno included.

**nontainer builds host objects per world** (`HostObject(factory=)`),
because one profile serves a session, its forks and its app. The host
half needs to know which world called: to name the helper's branch
under its caller, to ask through that world's `Sessions`, to build the
helper from that world's profile, and to stop with that world's turn.

### Helpers are delegates

**A helper is a delegate branch of its caller**, asked through the
calling world's `Sessions` (found by `Sessions.of(ws)`):
- **It lives where delegates live.** The branch is named under its
  caller (`lead.corners-1`), and a studio shows it, keeps it and sweeps
  it like any delegate.
- **It holds what its caller holds and no more:** the caller's profile,
  without `agex`, so helpers don't make helpers; `objects=` narrows it
  further. It sees none of the caller's files, and starts with no
  conversation.
- **An empty view forks from the last commit,** so a call made in the
  middle of `run_python` lands nothing of the caller's.
- **Each call brings its own runner** (`ask(runner=)`), holding the
  agent, the spec and the inputs, so the typed value never passes
  through nontainer: the runner keeps it, and the `Answer` carries a
  summary for anyone reading the job list.
- **The calling code collects its own answer,** inside the same call,
  so the caller's model is never handed it as a note.
- **A scratch world is the opt-out** (`agent_tasks(agent,
  scratch=True)`), and the fallback for a world with no `Sessions`
  open.
- **A cancel of the caller's turn stops its helpers** (`cancel_tasks`).
- **Fan-out is bounded:** `.map` runs at most `max_parallel` at once (4
  by default).

### Types cross as shapes

The host half never needs the caller's classes. It carries encoded
values between two sandboxes, and each has the real types:
- the stub compiles the signature in the kernel and sends it as data:
  a tree of kinds, with each record and enum given by name, fields,
  docstring and members;
- the host runs nothing the guest sent: it reads the spec, writes the
  brief, and passes encoded values through;
- the helper's world gets generated classes with the same names and
  fields, which its code builds and `task.success` checks against;
- the stub decodes the result into the caller's own classes, strictly.

So a class defined in a script works as one from a module does. The
helper doesn't get the class's methods or validators: constraints
written as types (`Literal`, enums) carry, and a check in a method runs
when the caller decodes. That is the usual trade at an RPC boundary.

**Values only, on every rung,** in-process included. A live object of
the caller's script would run as the caller's code in the helper's
context, and a live return would be the helper's code run later in the
caller's world. Agent-defined tasks take and return values, and behave
the same everywhere.

**What the agent can meet:**
- refused at the decorator, saying what to write instead: a lambda, a
  missing docstring or annotation, code in the body, a live type, a
  parameter named like a host object of the world;
- `agex.TaskFailed`, when the helper fails or its value fails the
  decode;
- `agex.TaskNeedsInput`, carrying the helper's question.

### The clock

**`run_python`'s timeout bounds the code, not the host it calls.** Time
spent in any host object's call doesn't count, on every rung: sandtrap
moves its checkpoint forward by the call's duration, and dud's
supervisor pushes its deadline back. The tick limit and the turn's
cancel still bound a script that waits. A stub's own methods are the
script's code and stay on the clock.

### In apps

Handlers run through `run_python`'s path, with the same host objects,
so `from host import agex` works there, and a task is defined at a
module's top level. Each call carries its spec, so nothing is
registered on the host.

- **The session's preview** runs against the session's world, so its
  helpers are the session's delegates once the session's helper is
  open, which its first turn does. Before that, they run on scratch
  worlds, which aren't listed, kept or swept as delegates.
- **A stand-in answers without a model** (`stand_in_tasks`): each call
  at once, with an embedder's `answer(name, inputs)` or the simplest
  value of the return type (the first choice of a `Literal` or enum).
  A run binds it in place of `agex`: `bind={"agex": "agex_stand_in"}`.
- **GET handlers are pure,** so tasks belong in POST handlers.
- **A published app** is a frozen snapshot with no `Sessions`, so its
  helpers go on scratch worlds. A request waits on its task.

## Delegation

`agent.session(ws, sessions=True)` gives the agent the `sessions` tool.
Each delegate is a fork of the session's world, run as a session of the
same agent by `agex.delegation.Runner`:
- **It holds what its parent holds:** it opens with the parent's
  profile, and can delegate in turn.
- **The runner is async,** scheduled on the parent's event loop rather
  than a thread each, so a delegate's model latency holds no thread.
- **Answers land as notes:** one that lands mid-turn rides the next
  tool result, and one that lands between turns waits for the next turn,
  or for `wake()`, a turn that opens with what is waiting.
- **`close()` waits for the delegates** the session started. A stop of
  the parent's turn doesn't stop them: they are sessions of their own,
  and an embedder that wants them stopped cancels them through
  `session.sessions` (the studio stops those its turn started).

nontainer owns the helper (`Sessions`), the branches and the answers;
agex supplies only the runner. Its public contract is nontainer's
`docs/sessions.md`.

## Tools

**What the model acts through is a tool; what agent code calls is a
host object.** The line is what the result is for:
- **a tool** when the model must see the result as such, or the
  embedder must see the call: pixels the model reads (`view_image`,
  `test_app`'s screenshots), a delegate's answer the person follows, a
  fork the embedder permits. `run_python`, `terminal`, `file_write`,
  `file_edit`, `test_app`, `view_image` and `sessions` are tools;
- **a host object** when code composes the result: `agex` tasks, an
  app's `db`, a studio's `web`.

**The tool protocol is MCP's.** A tool is nontainer's `Tool` and a
call's result its `ToolOutput`, both MCP-shaped (a name, a description,
a JSON Schema for the arguments; text, images and an error flag back),
so an MCP tool can become one and one can be served over MCP.

**An embedder shapes a session's tools:** a `Toolset` of its own (an
app runtime for `test_app`, primers, `vision`) and tools of its own,
one named like a built-in replacing it. A tool result's images reach a
model that takes images, through agex's record and provider.

## Compaction

nontainer owns the record of a fold and its rules (a `Fold` in the
`__compaction__/` plane, taken back by a rewind, carried by a fork,
never in a stored run; nontainer's `docs/compaction.md`). Each loop
does its own folding, and agex's (`agex.compaction`):
- folds before a model call that would reach the `Policy`'s budget,
  within a run too, so a long task folds its own earlier steps;
- never splits a tool call from its result;
- has the agent's own model write the summary, sent nearly the request
  it was about to send, so almost all of it is cached;
- counts summary calls against `max_steps` and in the run's usage.

## Providers

**pydantic-ai is the transport,** through `pydantic_ai.direct`; its
`Agent` is not used. agex keeps two seams of its own:
- **a small provider protocol** (`agex.providers.Provider`): `stream`
  the reply to messages and tools, say whether an error is `transient`
  (overloaded, rate-limited: worth a `resume`), and optionally
  `describe` a failed request in words;
- **its own run record,** mapped to and from pydantic-ai messages at
  the edge, so a third-party type never becomes the stored format.

`Settings` carries `thinking` (an effort, mapped per provider),
`max_tokens`, `temperature` and `extra`, passed through as they are.
`ScriptedProvider` replies from a conformance script, so tests run the
real loop without a model.

## Capabilities belong to the world

**The rule:** a world's environment is set by whoever opens it, and an
agent never changes it. Forks inherit it, delegates and helpers
included, whichever agent drives them. So capabilities are nontainer's
environment configuration, not agex's, and agex is just a loop.

- **No per-session overlay:** nothing is re-applied when another agent
  drives a fork.
- **Capabilities flow down, never up:** a fork can't gain what its
  parent lacks, and narrowing (`objects=`) only removes.
- **The cost:** a reusable agent ships as an (agent, profile) pair.

**What exists:** `nontainer.Profile`, the open-time configuration as one
value: module grants (`ModuleGrant`, and presets such as
`dataframes()`), host objects (live, or `HostObject` with a `type`, a
`stub` or a `factory`), the classes agent code builds, mounts,
commands, environment variables and the executor.

**Where it is headed** (the capabilities track in `docs/plan.md`):
- **`provide`,** code that runs in the kernel, compiled to module
  grants: recursive by default, with `unsafe=[...]` naming members that
  act outside the process, and a package's embedded skills mounted
  with it.
- **`HostObjectGrant(obj, include=, exclude=, scope=)`,** honoured by
  every transport, with scopes checked when a method is called: a
  locked call returns "needs input" (nontainer #143).
- **A generated host-surface index** in the tool text, one line per
  method, so a host object needs no hand-written primer. Today the
  tool text names host objects only.
- **`help()` on every transport.** In-process, `help(db)` shows the
  class; under process isolation and on dud it shows the proxy, whose
  marker carries method names only. The surface metadata should carry
  signatures and docstrings.

**Documentation is skills.** Old agex's visibility tiers and
`describe` overrides are gone: a skills catalog (frontmatter in
context, bodies read on demand), the host index and `help()` cover what
they did.

## From 0.12

| 0.12 | 0.13 |
|---|---|
| `fn` / `cls` / `module` registration, visibility tiers | the world's profile: module grants and host objects; skills for documentation |
| `state=`, `Live` / `Staged` / `Namespaced` | a nontainer `Store` and `Workspace` |
| `fs=`, `run_file_in_sandbox` | `ws.files`, `ws.run_python` |
| `host=` (local / http / modal) | gone: remote execution is an executor (dud) |
| isolation and resource limits on the agent | nontainer's executor and `PythonConfig` |
| `spawn`, `max_spawns` | agent-defined tasks and `.map`, or the `sessions` tool |
| `session=` on every task call | `world=` |
| `on_conflict` | gone: a task runs on a fork and can't conflict |
| `TaskClarify`, `PermissionPending` | `needs_input` and `resume` |
| `TaskTimeout`, `LLMFail` | `Outcome` statuses, raised when a plain call unwraps them |
| event classes, emissions | agex's run record (stored), nontainer's `TurnEvent` (streamed) |
| `agent_git` | nontainer's ws-git |
| `view`, token counts, `pprint_*` | `agex.debug`, not yet ported |

Two things were given up: a spawn could hand back a live object, and a
spawn was only a thread where a task call forks a branch and runs a
turn. Forks are O(1); the model call dominates.

## Curation

nontainer's `docs/curation.md` plans curators that keep a wiki of what
recurs across sessions and turn it into skills, libraries and services.
This design fits it: a curator pass is a task (`propose.run(pattern,
world=lib, keep=True)`, where a rejected proposal is a branch nobody
merged), a trial arm is a task on a fork the gate opens with a safe
profile (substitution by the opener, which the capabilities rule
allows), and a verdict is a task with a typed return. The library and
service split is `provide` and host objects: code the session owns
runs in its kernel; authority it shouldn't hold runs behind a host
object.

## Open questions

- Do server-side deployments stay on agno, or move to agex?
- A fork made in the middle of `run_python` commits the call's partial
  writes, splitting its commit in two. Acceptable?
- Where do host-object scope grants live: a reserved plane in the world
  (so a rewind would revoke them), or the embedder?
- Should skill metadata carry `kernel:` (python, ts), so each side
  skips skills it can't run?
- Should tasks declare `requires=[...]`, refusing a world that lacks a
  capability?
- A published app's task call holds its request open. Does it need a
  job shape (start, then poll), and budgets per app?
- On dud, should a terminal host-side error be marked so the guest
  can't swallow it? (The recorded value already wins.)
- Should agex connect to MCP servers, turning their tools into
  `Tool`s? Nothing blocks it; whether it fits "capabilities belong to
  the world" is the question.

## Decisions

- **2026-10-03:** agex-studio stays serverless and BYOK.
- **2026-10-03:** Python goes first: agex is rewritten as a loop over
  nontainer, and agex-ts follows the frozen shape.
- **2026-10-06:** agex depends on nontainer. Standalone would re-grow a
  world layer; inside nontainer would break "nontainer owns no model
  and no loop".
- **2026-10-06:** Capabilities belong to the world, not the agent.
- **2026-10-06:** Documentation is skills; visibility tiers and
  `describe` go.
- **2026-10-06:** pydantic-ai (`direct`) is the provider transport,
  behind agex's own provider protocol and run record. LiteLLM squeezed
  reasoning round trips, and agno's model layer would re-couple to agno.
- **2026-10-06:** The conversation plane is harness-neutral
  (`__conversation__/`): core owns a small index, run bodies stay in
  each harness's format. The turn is `with ws.turn(run_id)`.
- **2026-10-06:** Two corpora in one format: the harness corpus in
  nontainer, the shape corpus in agex. Scenarios are Python dataclasses
  with generated JSON, and the scripted model is the clock.
- **2026-10-06:** Each shape is declared once, in the lowest package
  that needs it: nontainer the turn and `TurnEvent`, agex the run
  record, `Outcome` and the task API, the studio its SSE events.
- **2026-10-06:** The open-time config is `nontainer.Profile`.
- **2026-10-07:** Harness host objects (attached after a world opens)
  are deferred: a task's fork gets `task` through
  `Store.fork(..., profile=...)`.
- **2026-10-07:** Task calls and task values as above: the declared
  type is the contract, in-process carries anything, every other rung
  exactly the encodable kinds.
- **2026-10-08:** The shape corpus is `agex.conformance`, with tasks
  written as JSON Schemas and a neutral model script.
- **2026-10-08:** nontainer keeps the record of a fold; each loop does
  its own folding.
- **2026-10-08:** Agent code defines tasks through the `agex` host
  object; code-level `ask` is dropped. Helpers are delegates, types
  cross as shapes, values only.
- **2026-10-09:** A host call's time is off `run_python`'s clock, for
  every host object.
- **2026-10-09:** A world's `agex` finds the world's helper through
  `Sessions.of(ws)`.
- **2026-10-10:** Tools are what the model acts through, host objects
  what code calls; the tool protocol is MCP's.
- **2026-10-12:** agex releases as 0.13.0 before the shape freeze, so
  the rebuild has users while async agent code lands. Host calls stay
  sync in Python agent code, awaited in TS.
- **2026-10-12:** A task call is typed by its return type only;
  arguments are checked at the call, and `world=` stays on it.
