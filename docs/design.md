# agex redesign

Status: direction agreed 2026-10-03; being built (see `docs/plan.md`).
Related:
- async agent code (top-level `await` in `run_python`, async host
  objects) has its own doc, `~/git/nontainer/scratch/async.md`;
- the staged implementation plan is `docs/plan.md`.

## Why

Two stacks carry the same core idea: **a versioned world that makes
quick-loop app authoring possible**. One kvgit branch holds files,
cache, cwd and the conversation; one commit captures all of it; the
agent acts by writing code and shell; forks are O(1); rewind is a
checkout; finished work leaves as a frozen subtree.

| | loop | code runs in | world | composition |
|---|---|---|---|---|
| agex (py, dormant) | agex | sandtrap | kvgit `Staged` + monkeyfs | typed calls, memoryless spawn |
| nontainer + agno | agno | sandtrap / dud VM | kvgit branch + VirtualFS | `Sessions.ask` fork, `Answer` names a commit |
| agex-ts / agex-studio | agex-ts, in the tab | Web Worker | kvgit-ts branch + KvgitFS (IDB) | typed tasks, memoryless spawn |

The two products have different fixed constraints:

- **agex-studio stays serverless.** Apps get published straight to
  tablets, and kids build their own apps with nothing but an
  OpenRouter key. No server is allowed in that loop.
- **Production use is server-side, Python, and built on nontainer.**
  nontainer-studio is mainly a proving ground for features headed
  there.

So there will be two codebases. The goal is to keep them aligned on
**meaning** (what fork, merge, delegation and outcomes do), not on
bytes or code. Divergent storage formats cost nothing until something
has to interoperate, and today nothing does.

## Vocabulary

- **World**: state. kvgit plus a file layout plus planes (files,
  cache, cwd, conversation, agent-git state).
- **Kernel**: where code runs. In Python that is the sandtrap
  in-process sandbox, a dud VM, or Pyodide; in TS it is a Web Worker
  or Node worker. nontainer calls this an `Executor`, and executors
  never commit.
- **Loop**: what drives the model. agno today; after this redesign,
  agex as a second option.
- **Host**: effects that leave the world (db, web, media, network).
  These are the only real side effects; everything inside the world
  is a value because it is versioned.

World, kernel and host are nontainer's. Only the loop is agex's.

The unit that crosses all four:

```
run(task, world=ref, inputs, policy) → events…, Outcome(ref′, value?, status, provenance)
```

Each existing concept is `run` with different defaults:

- **chat turn**: same branch; conversation carried; the value is text.
- **typed task**: a fresh fork; no transcript; the value is validated
  against a schema; the branch is garbage-collected unless kept.
- **delegate**: a fork with `inherit` / `paths`; the parent decides
  what to merge.
- **clarify / permission**: the outcome status is "needs input", and
  the run continues with `resume`.

## Top-level API (proposal, 2026-10-05)

**An agent is a loop, and the world gives it a body.** `Agent` holds
the model, the primer, loop settings, and a default environment that
is used only when the agent creates a world. Capabilities belong to
the world (see "Capabilities belong to the world" below).

Everything that runs a model is one `run`. There are two front doors,
and they differ only in their defaults:

| | session (side-effect) | task (functional) |
|---|---|---|
| world | the session's branch, mutated in place | always a **fork**; the caller's world is untouched |
| transcript | carried across turns | fresh |
| result | text, plus the new head | a typed value, plus the fork's ref |
| ends when | the model stops calling tools | `task.success(value)` |

So "functional" is precise: a task is pure with respect to the world,
and its only effects go through host objects.

```python
profile = nontainer.Profile(
    provide=[pandas, plotly],
    host={"db": HostObjectGrant(db, include=["query"], scope="db")},
    skills=["skills/flashcards"],
)
ws = store.open("kid-app", profile=profile)

agent = Agent(model=..., primer="You help build study apps.", profile=profile)

# side-effect shape
chat = agent.session(ws)                     # the world's own environment
reply = chat.say("make a flashcard app")     # Turn(text, head, events)
async for ev in chat.stream("add a timer"): ...
chat.inbox.put("make it purple")             # mid-run steering

# functional shape
@agent.task
def grade(responses: list[Response]) -> Report:
    """Grade the quiz."""

report = grade(rs)                           # scratch world built from agent.profile
out = grade.run(rs, world=ws, keep=True)     # fork of ws; Outcome: .value .ref .status .events

# inside agent code: side effects through the sessions tool, functions as tasks
@agex.task
def review(path: str) -> TestReport:
    """Review the tests under path."""

reports = review.map(["app/", "lib/"])
```

Each front door has a sync and an async form (`say` / `asay`, plus
sync or `async def` tasks). The loop underneath is async-native.

### Outcomes come back through a host object

`task.success(value)`, `task.fail(reason)` and
`task.needs_input(question)` (covering both clarifying questions and
permission requests) are methods on one `task` host object. It is a
*harness host object*: the loop attaches it to each session it runs.
nontainer needs no knowledge of agex. A task's fork gets it through
`Store.fork(..., profile=...)`, which adds host objects when the fork
opens.

**Probed on nontainer 0.8.7, in-process and process isolation:**
- A bare function works as a host object.
- When the handler's validation fails, agent code gets a `TypeError`
  at the call site. The agent can fix it within the same script. Old
  agex only validated after the iteration.
- The handler records the value, then raises a `BaseException`
  subclass. The script stops immediately, and agent code's
  `except Exception` doesn't swallow the signal.
- Files written before the call are kept and committed. The stop
  signal shows up as `result.error`, and the loop must not show it to
  the model as an error.

**How the loop uses it:**
- **It trusts the recorded value, not `result.error`.** It checks the
  slot after every call. Stopping early is a per-transport nicety. On
  dud (from code reading), a host-side error reaches the guest as a
  plain `RuntimeError`, which `except Exception` could swallow; the
  recorded value still wins there.
- **One object per session, with a swappable slot.** The loop sets the
  slot (the task's spec, the destination) before each run. The
  single-writer rule makes the swap safe. Delegates are their own
  sessions with their own objects.
- **How the value travels depends on the rung**, and the declared type
  decides what may travel at all. See "Task values" below.
- **The value is also written to the world.** The handler writes the
  encoded value under a reserved key, through the workspace's
  re-entrant lock (host objects calling back into the workspace is a
  tested invariant). It lands in the same commit as the call, so a
  resume, or a parent reading a delegate's ref, can find it.
- **sandtrap cleanup.** sandtrap can drop its leftover agex-specific
  `TaskSuccess` / `TaskContinue` handling.

### Task calls (agreed 2026-10-07)

- **What agent code sees:** each argument bound by its own name, the
  way a function's parameters read, plus `task`.
- **Statuses.** `Outcome.status` is a run status or one of two
  task-level ones:
  - `success`, with `value`;
  - `failed`, from `task.fail(reason)` or an error;
  - `needs_input`, with the question; see "Resuming" below;
  - `cancelled`;
  - `interrupted`.

  A plain call (`grade(rs)`) returns the value, or raises `TaskFailed`
  or `NeedsInput`.
- **A model that stops without finishing** is nudged within the same
  run to finish with `task.success(...)` or `task.fail(...)`, up to
  twice. Then the task fails.
- **Worlds:**
  - by default, a scratch world in a memory store, built from
    `agent.profile` and discarded afterwards;
  - `world=ws` always forks, through `Store.fork(..., profile=...)`
    with `task` and the inputs added;
  - `keep=True` keeps the fork, and `out.ref` names it.
- **Resuming.** A `needs_input` outcome always keeps its fork,
  whatever `keep=` says, and `out.ref` names it.
  - `grade.resume(ref, answer, world=ws, **live_inputs)` continues it
    by ref, through the world the task ran on (for a scratch world,
    without `world=`, in the same process). Encodable inputs come back
    from the `__task__` plane. Live inputs are supplied again and bound
    as host objects; a missing one is refused by name.
  - `out.resume(answer)` is the in-process shortcut, reusing the
    outcome's own inputs.
  - **Surviving a restart needs a persistent world.** A scratch world
    lives in a memory store, so it resumes only within the process. A
    request approved next week runs with `world=` on a disk or
    Postgres store.

### Spawn folds into delegation

| what spawn gave you | replacement |
|---|---|
| memoryless clone | `@agex.task` from agent code, run on a scratch world (see "Agent-defined tasks") |
| typed result | the task's return type |
| fan-out (`submit` / `map`) | `.map`, bounded on the host |
| `max_spawns` | the `agex` host object's concurrency limit |
| clone events streamed but never stored | helpers are delegate branches: stored, shown and swept like any delegate |
| depth-1 limit | the same by default: a helper's world has no `agex` |
| work in a world, reported back | the `sessions` tool: delegates are real sessions, stored, open-able, resumable |

The old dual-decorator "agents as functions" folds in the same way: a
task's worker is the agent the embedder put behind `agex`, and its
world holds what its caller's holds and no more.

There are two real losses:
- **Live in-process return values** off in-process. A spawn could hand
  back a closure or a live object. A task's value crosses through the
  value encoding ("Task values"), and in-process still carries
  anything.
- **Cheapness.** A spawn was just a thread. A task call forks a
  branch and runs a turn. Forks are O(1); the model call dominates.

### What stays

- `Agent`: model, primer, loop settings (budget, chaptering), and a
  default `profile` for worlds the agent creates.
- `@agent.task`: typed inputs, a validated typed return, and a retry
  on mismatch.
- `agent.session`, `Outcome`, and the event stream.
- Permission requests. The scope itself lives on a nontainer host
  object. agex turns a locked call into a "needs input" outcome and
  continues it with `resume`, so a request can be approved next week.
- Chaptering, built on `nontainer.compaction` (released in 0.8.8),
  after curation's trace projection lands (plan step B4).
  Its records are harness-neutral, and `docs/compaction.md` reserves
  chaptering for later:
  - a chapter is a fold record with `first` set, plus a name;
  - the agent's own model writes the summary, using the agex chapter
    prompt;
  - the originals stay readable as files, via curation's trace
    projection.

  agex folds its own conversations, writing nontainer's fold records:
  nontainer owns the record and its rules, and each loop its folding
  (decided 2026-10-08).
- `agex/bench`, as `agex.bench`, off the core surface.

### What goes

| agex surface today | fate |
|---|---|
| `fn` / `cls` / `module` registration | nontainer environment: `provide` and `host` |
| `visibility` tiers, docstring overrides, `capabilities_primer`, `MemberSpec` configure | skills, plus nontainer's generated host-surface index |
| `state=`, `Live` / `Staged` / `Namespaced`, `StateResolver`, `connect_state`, `events(state)` | nontainer Store / Workspace |
| `fs=`, `connect_fs`, `run_file_in_sandbox` | nontainer (`ws.files`, `ws.run_python`) |
| `host=`, `Host` (local / http / modal), `clear_agent_registry` | gone; remote execution is an Executor (dud), deployment is the embedder's |
| `isolation`, `max_memory_mb`, `max_open_files`, `eval_tick_limit`, `eval_timeout_seconds` | nontainer (the environment and the executor) |
| `spawn`, `max_spawns`; TS `spawn`, `maxSpawns`, `captureSpawnEvents` | delegation |
| `session=` string kwarg on every task call | gone; an explicit `world=` |
| `on_conflict` / `max_conflict_retries` | gone; a task runs on a fork and can't conflict, and merging back is explicit |
| sub-agent failures squashed into tracebacks | `Outcome.status` from a delegate |
| `agent_git` | nontainer ws-git |
| emission classes, the nine event classes | stored: agex's own run record (curation's `TraceSource` renders it; nontainer keeps stored formats adapter-owned, so agno's runs never migrate). Streamed: nontainer's `TurnEvent`, shared by every harness and the studio |
| `TaskClarify` (ends the task), `PermissionPending` | one "needs input" status plus `resume`; clarify becomes resumable |
| `TaskTimeout`, `LLMFail` | `Outcome` statuses (`capped`, `failed`), raised as exceptions when a plain call unwraps them |
| `agex_primer_override`, `setup=` on tasks | probably gone; setup can ride the first message |
| `view`, token counts, `pprint_*` | `agex.debug` |

The TS side mirrors this at the shape freeze:
- `createAgent` keeps the loop and moves capabilities into the TS
  world package's environment;
- `agent.task`;
- a new `agent.session(world)`;
- no `spawn`.

## Task values (2026-10-07)

How a task's inputs and its value travel between the caller, the
kernel and the world, on every rung.

**JSON Schema can't be the contract.** It describes JSON values, so it
covers only the data subset of Python types. Checked on pydantic
2.13:
- plain classes, DataFrame-like objects and callables have no schema;
- `Union[Response, Frame]`, with `Frame` an opaque class, yields
  `Response`'s schema alone. The other branch is dropped without a
  warning.

### What crosses today

- **Into the kernel is the safe direction.** The kernel only unpickles
  what the host wrote. dud already sends app handlers' `Request` in as
  a host-to-guest pickle.
- **Out of the kernel is the risky one:**
  - in-process: there is no boundary, so the caller gets the live
    object;
  - process isolation: the host unpickles it, which is the hole in
    sandtrap's own threat model;
  - dud: the host never unpickles. Only JSON, bytes and file refs
    cross, and large payloads are expected as files.
- **Variables don't survive between `run_python` calls.** Task inputs
  are bound again on every call, as host objects already are.
- **nontainer keeps per-call data apart from live resources**
  (`inputs` vs `host_objects`). Merging them would make moving off
  in-process "a silent breaking change"; keeping them apart "makes the
  contract checkable at the right moment". Task values follow the same
  principle.

### Rule 1: the declared type picks the encoding and drives decoding

Pickle's flaw is that it rebuilds whatever the bytes ask for. Here the
decoder only ever builds the type the task declared, never what the
payload claims. Checked on pydantic 2.13:
- in-process, validating a valid value returns the same object;
- JSON decoded by the declared type rebuilds sets, tuples, datetimes
  and enums exactly.

| kind | examples | on the wire | JSON schema | where it can go |
|---|---|---|---|---|
| data | primitives, containers, dataclasses, TypedDict, pydantic models, Enum, datetime | JSON | yes | everywhere, including apps and TS |
| table | DataFrame, Series, `pa.Table`, anything with `__arrow_c_stream__` | Arrow IPC stream | columns only | wherever pyarrow (or apache-arrow in TS) is |
| array | `ndarray` | `.npy`, loaded with `allow_pickle=False` | no | any Python rung |
| bytes and files | `bytes`, a file the agent wrote in its world | bytes or a file ref | yes | everywhere |
| live | callables, clients, generators | none | no | in-process only |

- **Nested values work.** On the wire a value is a JSON tree whose
  leaves can be tagged references to binary parts, and a tag is
  accepted only where the declared type allows that kind. So
  `dict[str, DataFrame]` comes back as a dict of DataFrames. (App
  handler returns flatten nested tables to rows instead, which is an
  HTTP choice.)
- **Embedders can register more kinds.** A kind needs an encoder that
  can run in the kernel and a decoder on the host.
- **`Any` or no annotation** means any data off in-process, decoded as
  plain JSON values.

### Rule 2: two classes of world, checked early

**In-process carries anything. Everything else carries exactly the
encodable kinds.** "Everything else" is process isolation, dud, apps,
TS, and anything stored for a later resume.

**Process isolation uses the encoding too, not a restricted
unpickler.** sandtrap's roadmap leans toward a restricted unpickler
("option A"). That would let process isolation carry more than dud, and
moving a world from process to dud would become a breaking change. With
one encoding, "works under process isolation" implies "works on dud".
The `task` channel then carries only built-in values, so a restricted
unpickler needs nothing beyond the built-ins on that path.

**The check runs as soon as the rung is known, and never mid-run:**
- when a task is called, against the world it will run in (the scratch
  world from `agent.profile`, or `world=`), before any model call;
- in the kernel, when agent code defines a task there.

Applying `@agent.task` checks nothing: a task whose types only an
in-process world carries is still valid with an in-process `world=`,
whatever `agent.profile` says.

The error names the type, the rung and the ways out:

```
TypeError: clean returns Callable[[Row], bool], which only an in-process world
can carry (this world runs under isolation="process"). Return data, a table or
a file instead, or open the world with isolation="none".
```

**Validation is by the declared type, the same on every rung.** The
caller's inputs are checked strictly, as the Python values they are.
The value agent code passes to `task.success` is encoded and decoded
by the return type, in-process too, so what reaches the caller is
built afresh as the declared types, whatever the agent passed. That
makes the encoding's looseness the rule everywhere: a list passes for
a tuple or a set, and a dict with a record's fields (or a record of
the agent's own with them) for the record. `"42"` is refused for an
`int` either way.

### Inputs

**Inputs are values on every rung.** A task never changes what its
caller passed, in-process included. Each input is typed host data
(`HostObject(value, type=...)`), so the task's code gets a fresh copy
each run, as a worker's unpickled one is:
- **in-process:** `nontainer.values.copy`. Data and pandas tables are
  deep-copied and arrays copied: a read-only view can be made writable
  again, and a copy-on-write table's `to_numpy()` hands back the shared
  buffer, so neither keeps the caller's object safe. Arrow and polars
  tables, whose buffers refuse writes, pass as they are;
- **off in-process:** pickled in, host to sandbox, the safe direction.

A change the task's code makes to an input lasts the run it was made
in; a value it wants to keep, it binds to a name of its own.

**A live input is a capability, not a value.** A client or callable
passed as an argument is bound as a host object of the task's fork,
which the caller opens, so it goes through the same policy as any host
object. Wrapping the argument narrows it:
`clean(df, db=HostObjectGrant(db, include=["query"]))`. That is the
opener setting the environment, which "Capabilities belong to the
world" allows. Agent code can't pass live values to the tasks it
defines (values only, on every rung; see "Agent-defined tasks"), so a
helper gains nothing this way.
- **Large values spill to blobs in the reserved `__task__` plane,
  never to files in the agent's tree.** curation's rule ("no agent's
  tree has a second author") keeps loop-written data out of the file
  tree. The transport for large blobs is the nontainer step's to
  settle: dud already moves large cache values on binary frames, and
  its hostcall payloads are specced small.
- **A live input can't be stored**, so resuming after a restart needs
  the embedder to supply it again (see "Resuming").

### The `__task__` plane

A reserved plane, like `__conversation__`, holding the task's spec,
its encoded inputs and its encoded value. A resume, a parent reading a
delegate's ref, or an app reads it there. A live value is not stored;
the `Outcome` holds it.

```
__task__/spec            name, instructions, each type (and its schema), which inputs are stored
__task__/inputs/<name>   an input sent by value, encoded (nt-value/1)
__task__/state           running, success, failed (with the reason) or needs_input (with the question)
__task__/value           the value, encoded, when it can be stored
```

The task's own loop writes it, so each write lands in the commit of the
run it belongs to: a world at any commit says how its task stood there.
A task starts its plane afresh, so a world forked from another task's
world doesn't carry that task's plane into its own.

### What the model sees

Python, never a schema:
- the signature;
- the source of the types involved;
- a preview of each input (a table's columns and first rows, an
  array's shape and dtype);
- for a live input, how agent code uses it: a function by its
  signature and docstring summary, any other object by its class's
  docstring summary and public methods, the class's own first.

Old agex did the same. The model writes Python anyway.

### Who defines tasks

**People embedding agex** annotate normally. Data and tables need no
registration.

```python
@agent.task
def clean(df: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    """Apply the rules to the table."""

clean(df, rules)                          # in-process: anything, live objects
clean.run(df, rules, world=isolated_ws)   # Arrow for df, JSON for rules
```

`Outcome.value` is always the declared type, never a dict or bytes
stand-in. If it can't be, the check fails before the run.

**Agent code** defines tasks through the `agex` host object (see
"Agent-defined tasks"), never the embedder's `Agent`:
- an `Agent`'s public attributes reach the model client and the
  embedder's full profile, so a scratch world built from it could hold
  grants the caller's world lacks;
- a bare `Agent` doesn't know which run called it: no shared budget or
  depth, no cancel from the parent, no events in the parent's stream;
- a script has no `ws` to pass. The host half knows the world it
  serves, and builds the helper's world from that world's profile.

An agent-defined task's types cross as shapes, so a class defined in a
script works as one from a module does.

**Apps** carry the data, table and bytes kinds. A data type's schema
feeds the contracts generated from an app's handlers
(`scratch/reuse.md`) and TS types, and Arrow is already the apps'
table format. Calling a task from a handler raises its own questions,
listed under "Open questions".

### How it fits nontainer and dud

| piece | what it is |
|---|---|
| a value encoding in nontainer | Generalizes the handler-returns encoder (`nt__Encoder` in `nontainer/apps/contract.py`), which already encodes in the worker, ships to a dud guest as source, imports numpy, pandas and pyarrow lazily, and does Arrow. #110's allowlist becomes a special case, since `Response` is data. nontainer is the lowest package that needs it. |
| a kernel stub plus a host half | The kernel side encodes before the host call; in-process it passes the value through. `task`, the handler encoder and a kernel-side task decorator all need this. Today it is done ad hoc; it could become a grant option. |
| dud | Unchanged. Its JSON, bytes and file values map one-to-one onto the wire above. |
| the `__task__` plane | A reserved plane next to `__conversation__`. |

## Agent-defined tasks (2026-10-08)

Delegation comes in two shapes, and each has its own door:
- **Side effects** (do work in a world, share files, report back in
  text) go through the `sessions` tool. B5a shipped it in agex, the
  same tool agno's sessions already have.
- **Functions** (a typed value from inputs) go through code. Agent
  code defines a task the way an embedder does, and calls it from a
  script, a helper module or an app.

Putting `sessions` into Python would give code a second door to the
side-effect shape, and no workflow was found that needs it. So
code-level `ask(returns=, agent=)` is dropped in favour of tasks.

### What agent code writes

```python
@dataclass
class Point2D:
    x: float
    y: float

@agex.task                        # or @agex.task(primer="You do geometry.")
def corners(shape: str) -> list[Point2D]:
    """The corner points of the named shape, counterclockwise from the origin."""

pts = corners("unit square")      # its own Point2D objects, or TaskFailed
many = corners.map(["square", "hexagon"])   # run at once, results in order
```

This is old agex's dogfood example (an architect building `Agent()`
plus `.task` from code) in the new shape.

### A host object, not a primitive

`agex` is a stubbed host object (`HostObject(..., stub=...)`, like
`task`). An embedder grants it, and the name is the embedder's choice.
The built-in tools stay `run_python`, `terminal` and the file tools.

- **Agent code holds only the stub.** The host half holds an `Agent`
  the embedder supplies (model, primer, limits), and agent code never
  reaches it. That answers the reasons for not handing agent code an
  `Agent` (see "Who defines tasks"): no model client, no embedder
  profile, nothing to configure past a primer.
- **Any harness on nontainer can grant it,** agno included, since host
  objects are nontainer's. The worker is an agex agent, because "a
  spec in, a typed value out, by running code" is how agex runs tasks.
  It reaches nontainer as a `SessionRunner` the call brings with it
  (see "Helpers are delegates"), so a second worker would be one more
  runner, not a new protocol.

### One host object per world

A profile is fixed when a world opens, forks inherit it, and a
host-object call doesn't say who is calling. One `agex` object in a
profile would serve the lead, every delegate and every helper, without
knowing which one called. The host half needs to know: to name the
helper's branch under its caller, to ask through that world's
`Sessions`, to build the helper from that world's profile, and to stop
with that world's turn.

**So nontainer builds host objects per world.** A `HostObject` can be
given a factory, called with each world as it opens (forks included):

```python
profile = Profile(python=PythonConfig(host_objects={
    "agex": HostObject(factory=lambda ws: Tasks(agent, ws), stub=TasksStub),
}))
```

The factory decides by the world it is handed, which is what lets one
host object fit a session, its app's preview and a published app (see
"In apps"). Other host objects that need their world can use it too.

### Helpers are delegates

**By default a helper is a delegate branch of its caller**, asked
through the calling world's `Sessions`:

```python
# in the host half, per call
job = sessions.ask(
    "corners(shape='unit square')",      # what the rail shows
    paths=[],                             # an empty view, forked from the last commit
    runner=TaskRun(agent, spec, inputs),  # this job's runner
)
```

- **It lives where delegates live.** The branch is named under its
  caller (`lead.corners-1`), and the studio's rail shows it, keeps it
  and sweeps it like any delegate.
- **It holds what its caller holds and no more:** the caller's
  profile, without `agex` unless the embedder allows a deeper level.
  It sees none of the caller's files, and it starts with no
  conversation.
- **An empty view forks from the last commit.** A fork lands the
  caller's uncommitted writes first, which mid-`run_python` splits
  that call's commit. A child that sees no files doesn't need them, so
  with `paths=[]` nontainer forks from the last commit and lands
  nothing.
- **Each ask can bring its own runner.** A world's `Sessions` runs its
  jobs on one runner, which under agno is agno's. A task's job runs on
  the `TaskRun` the host half made for that call, holding the agex
  agent, the spec and the inputs. Without `runner=`, the helper's own
  runner is used, as today.
- **The typed value never passes through nontainer.** `TaskRun` keeps
  the encoded value, and writes it to the child's `__task__` plane as
  tasks already do. `Answer.text` is a summary for the rail and for a
  model reading the job list.
- **The calling code collects its own answer.** The host half reads it
  with `result()`, which marks it collected, inside the same
  `run_python` call, so the caller's model is never handed it as a
  note. A script that dies before collecting leaves the answer to be
  delivered as any other.
- **Nothing rolls up, as for delegates.** A helper's events and usage
  are its own runs' on its own branch. The calling run gets the value.
- **The opt-out is a scratch world** in a memory store, built from the
  caller's profile, for an embedder that doesn't want helpers kept.

### Limits

- **The model is the embedder's agent's.** Code picks a primer, not a
  model. An allowlist of agents can come later.
- **Fan-out is bounded.** `.map` runs calls at once on the host, up to
  a limit the embedder sets.

### Types cross as shapes

The host half never needs the caller's classes. It carries encoded
values between two sandboxes, and each has the real types:

- **The stub compiles the signature in the kernel.** It reads the
  name, the docstring and the annotations, resolved with the defining
  code's names, and sends a spec: the signature as a tree of kinds
  (`values.Spec`, exported as data), with each record and enum given
  by name, fields, docstring and members.
- **The host runs nothing the guest sent.** It reads the spec as data,
  writes the brief from it, and passes encoded values through
  undecoded.
- **The helper's world gets generated classes,** a dataclass per
  record and an `Enum` per enum, with the same names, fields and
  docstrings. Its code builds them, and `task.success` checks against
  them, so a bad value is fixed within the helper's script.
- **The stub decodes the result by the caller's own annotation,**
  strictly. The caller gets its own classes back, with their methods
  and validators.
- **What a shape gives up:** the helper doesn't get the class's
  methods or custom validators. Constraints written as types
  (`Literal`, enums) carry; a check in a method runs only when the
  caller decodes, and a value it rejects ends the call `TaskFailed`.
  That is the usual trade for an RPC boundary. The helper makes data,
  and the caller owns behaviour.

This replaces the rule that an agent-defined task's types must come
from a module in the world: a class defined in a script works the same
way. Classes as shapes also make source recovery unnecessary, with no
module files to copy and no imports to follow.

**Values only, on every rung.** The full encodable set (data, records,
enums, bytes, tables and arrays) crosses, and no live object does, in
process included. An embedder's task keeps live inputs in-process,
where the embedder hands over a real client and narrows it; agent code
has nothing to hand over that way:
- **a host object it holds** is one its helper inherits already, but
  `agex`, which the helper lacks on purpose;
- **a function or object of its own script** would run, called from
  the helper, as code compiled in the caller's sandbox, with the
  caller's globals, in the helper's context;
- **a live return**, a closure say, would be code the helper's model
  wrote, run later in the caller's world.

So agent-defined tasks take and return values, and behave the same on
every rung. A workflow that needs more can bring a narrow rule later
(forwarding a host object the caller holds, never `agex`, say).

**Inputs share a namespace with the world's host objects.** Each input
is bound by its own name, as a host object of the helper's world, and
the helper inherits its caller's host objects, so a parameter can't be
named like one of them. The decorator checks this against the caller's
host objects, so the clash is refused where the task is defined.

### What the agent can meet

- **Refused at the decorator, saying what to write instead:**
  - a lambda;
  - a missing docstring or annotation;
  - code in the body, when the source is readable;
  - a live type in the signature, on every rung;
  - a parameter named like one of the world's host objects.
- **`TaskFailed`** when the helper fails, or its value fails the
  caller's decode.
- **`TaskNeedsInput`**, carrying the question, when the helper asks.
  No caller model is there to answer it, so it is an error the calling
  agent reads. Resuming waits until something needs it.

### The clock

A host call counts against `run_python`'s timeout on every rung today.
Probed with a 2-second host call under a 1-second timeout: sandtrap
times out in-process (2.0s) and under process isolation (2.1s), and
dud's guest supervisor kills the run (2.3s). A few helper calls would
use up the caller's script.

- **Old agex never paused the clock.** Its main guard was sandtrap's
  tick limit, which counts executed code, so it can't advance while a
  host call waits. It raised the wall clock to 300 seconds whenever a
  tick limit was set, "so sub-agent LLM calls don't trigger a timeout".
- **Here, the timeout bounds the code, not the host it calls.** Time
  spent in any host object's calls doesn't count (decided 2026-10-09,
  replacing an opt-in per host object):
  - sandtrap moves its checkpoint's start time forward by the call's
    duration, in-process and in the process worker;
  - dud's supervisor pushes its deadline back by the time it spends
    relaying the call.
- **The tick limit and the calling turn's cancel still bound a script
  that waits,** so a script looping on a slow host function ends by
  ticks or by cancel rather than by the clock. A stub's own methods
  are the script's code and stay on the clock; only its calls through
  `remote` are the host's.

### In apps

Handlers dispatch through `run_python`'s path, with the same policy and
host objects, so `from host import agex` works there, and a task can be
defined at a module's top level:

```python
# app/tasks.py
from host import agex

@agex.task
def summarize(notes: list[Note]) -> Summary:
    """Summarize the notes for the dashboard."""
```

The decorator runs when a handler's kernel imports the module, and a
kernel can be made again, so registering a spec is idempotent (keyed
by the spec).

- **The session's preview** (the live preview, `test_app`, `ws-curl`)
  runs against the session's own world, so its helpers are delegate
  branches of the session, as from `run_python`. No turn is running,
  so there is no turn to stop with, and a person clicking spends the
  model calls. `bind=` gives a stand-in: `bind={"agex":
  "agex_scripted"}` runs the page against a scripted worker, and
  ws-pytest's `call(..., agex=...)` does the same for a test.
- **GET handlers are pure** (a read-only filesystem and cache). A task
  call spends and makes a branch, so tasks belong in POST handlers.
- **A published app** is a frozen snapshot with no session, no
  `Sessions` and nothing to write to. The factory sees that, and puts
  helpers on scratch worlds, or in a namespace the embedder picks for
  the deployment, with its own sweep. Whether a deployment gets
  `agex` at all, and with what budget, is the embedder's, at the open
  (see "How do app handlers call tasks?" under "Open questions").
- **A request waits on its task.** The clock rule covers handlers too,
  but a long task wants a job shape (start, then poll), which is part
  of the same open question.

### Found by building it (B5b-1)

- **A world's helper registers itself.** The harness builds its
  `Sessions` after the world opens, so the host half finds it at the
  call: `Sessions.of(ws)` is the helper open over a world (nontainer
  #228). No harness changes, and the studio's helper is found the
  same way as agex's.
- **Generated classes needed a way across.** The classes
  `load_specs` builds have no module, so a worker process couldn't
  import them and a dud guest couldn't rebuild them from source.
  nontainer #229 makes them pickle as the data they were built from,
  and each rung builds them again from it.
- **Forks are made on the calling thread.** A host call runs on the
  thread running the caller's script, which holds the world's lock,
  and a fork takes it again. `.map` forks each helper there, then
  waits for them together; forking from other threads deadlocked.
- **A class with no source is written out for the brief** from what it
  holds: `@dataclass`, docstring, fields with defaults, and enum
  members, so the helper's model sees what the caller wrote.

### Found by the spike

The spike gave a live model (Claude Haiku 4.5) an `agents.task(fn,
primer=)` host object, in-process. It built a linear-equation solver
and called it three times; each call returned the right float, and the
helpers took about half of the 22-second turn.

- **Host-side async work inherited the sandbox's context.** sandtrap's
  in-process sandbox denies the network and redirects the filesystem
  through context variables, and a coroutine scheduled from inside a
  host call copied them. Every helper failed its model call with
  "Connection error" until the work ran in a clean
  `contextvars.Context()`. Any host object that calls a model or the
  network asynchronously meets this, so the fix belongs on nontainer's
  in-process host-call path.
- **The model reached for a lambda first,** which has no signature or
  docstring to make a spec from. It failed late with an invalid world
  name. A refusal at the decorator fixes the first try.
- **Calls in a loop ran one after another.** `.map` is the fan-out.
- **On dud, a dataclass defined in `run_python` fails** (dud #40). The
  guest runs code in a dict named `__dud__`, no module of that name is
  in `sys.modules`, and `dataclasses` looks it up there. Agent-defined
  records need it fixed.

## Capabilities belong to the world (proposal, 2026-10-06)

**The rule:** a world's environment is set by whoever opens it, and an
agent never changes it. Forks inherit it. That includes delegates,
whichever agent drives them. The registration API is therefore
nontainer's environment configuration, not agex's.

**What this buys:**
- **No per-session overlay.** Nothing needs re-applying when a
  different agent drives a fork.
- **Capabilities flow down, never up.** A fork can't gain anything its
  parent lacks. Narrowing a child later, e.g.
  `ask(..., hosts=["web"])`, is safe because it can only remove.
- **The registration work benefits everyone.** It lands in nontainer,
  so agno users and server-side deployments get it too.
- **agex shrinks to a loop.**

**What it costs:**
- A reusable agent ships as an (agent, environment) pair.
- A task can declare `requires=[...]` and refuse to run in a world
  that lacks it.
- Old agex's `@agent.fn` decorator style becomes sugar on the
  environment builder.

### Profile shape

`nontainer.Profile` (built in A0) is a small frozen bundle of today's
open-time config: python grants and host objects, mounts (skills ride
in as mounts), commands, environment variables, the executor factory,
the root and the ignore patterns. It exists so the environment can be
passed around as one value: to `store.open`, as an agent's default,
and down to forks.

**`provide`: code that runs in the kernel.** It compiles to
`ModuleGrant`.
- `recursive` is the default, as nontainer's presets already do for
  numpy, pandas, pyarrow, matplotlib and plotly. On dud and TS the
  whole package is simply there.
- `unsafe=[...]` names members that act outside the process. It is a
  fact about the library, not a policy:
  - in-process, those members are gated;
  - on dud the fact is still true but harmless, since the guest has no
    network and its only exits are host objects, so nothing needs to
    happen there;
  - known libraries bring the presets' curated lists (`provide(pandas)`
    picks up `_PANDAS_EXCLUDE`), so authors only write `unsafe=` for
    their own modules.
- Effects with credentials (an `admin_*` that reaches production) are
  not `unsafe` members. They are host objects, or they are left out.
- A provided package's embedded skills (`<pkg>/skills/<name>/`) are
  mounted read-only automatically. Making a library available and
  teaching its use become one gesture. Mounting rather than
  `install_from_modules`'s copy means no commit noise, and the skill
  tracks the installed version. The turn commit's profile
  fingerprint records that version (see "Fit with curation").
- There is no sandtrap inside dud. Running it there would buy no
  security (the VM is the wall), and real bash in the guest would
  bypass it anyway.

**`host`: objects and functions that run on the host.** This is
authority, enforced on every kernel.
- `HostObjectGrant(obj, include=, exclude=, scope=)` is honored by
  every transport:
  - in-process, by the sandtrap policy;
  - under process isolation, by the RPC handler and its declared
    surface;
  - on dud, by the hostcall allowlist.

  Today in-process registers the default patterns, while process
  isolation and dud expose every public method (`apps.md` lists "no
  `HostObjectGrant`" as a known gap).
- Scopes are checked when a method is called, on the host side. A
  locked call returns "needs input" instead of requiring a config
  change, which works on every transport. This connects to nontainer
  #143, the visibility and ask-permission hook.
- **A generated host-surface index** goes into the tool text, one line
  per method: `db.query(sql: str) -> list[dict]: Run a read-only
  query.` It is derived from the grant plus signatures and first
  docstring lines, so it can't drift. Today nontainer lists only host
  object *names* (`render.py` ~688), and the studio compensates with
  hand-written primers.
- **`help()` works on every transport.** `RpcProxyMarker` carries
  method names only, so under process isolation (and on dud)
  `help(db.query)` shows the proxy. The surface metadata should carry
  signatures and docstrings.

**Documentation is skills.** `visibility` and `describe` are dropped.
- The skills catalog, i.e. frontmatter in context with bodies read on
  demand, plus the host index and `help()`, cover what visibility
  tiers did.
- "Low visibility" just means no skill.
- Skills that are fixed by the environment (a package's docs, a
  service's generated contract) are read-only mounts at
  `skills/<name>` (`nontainer.skills.mounts`).
- Skills that belong to the world are ordinary versioned files that
  the embedder installs or the agent writes. That includes every
  skill that evolves: curated releases are vendored at a version, as
  `docs/curation.md` specifies.

**Mounts stay a world concern.** A data mount is about which world
this session sees. agex passes them through; it doesn't wrap them.

**The TS kernel has one gap.** A URL import maps to `provide`, and a
live reference to `host`. But the worker walls off the DOM, not
`fetch`, so an `unsafe` member doing network I/O isn't contained
there. The kernel should report that when it opens.

**What agex still needs from nontainer:**
- a default profile for the worlds it creates (`Agent(profile=...)`);
- harness host objects attached per session (plan step 1).

## Fit with curation (2026-10-06)

`nontainer/docs/curation.md` is the far-fetched plan:
- a maintainer keeps a wiki of what recurs across sessions;
- a builder turns the wiki into skills, prose and code with tests;
- a gate runs paired trials and probation;
- sessions install the releases.

The hope reaches further still: curators that write libraries or
whole virtual backend services, taught to other sessions through
skills generated on the fly. The redesign has to keep those doors
open. It does, and in places it fits better than the design
curation.md was written against.

**The gate and the curator passes are tasks; workers are sessions.**

| curation piece | new shape |
|---|---|
| a builder pass: fork `lib`, make one proposal commit | `propose.run(pattern, world=lib, keep=True)`. The outcome's `ref` is the proposal branch and its `value` a typed `Proposal`. A task always runs on a fork, so "a rejected proposal is a branch nobody merged" comes for free |
| a trial arm: a fresh start from a real state, skills taken in | `task.run(prompt, world=fork_spec)`. `world=` accepts a fork spec (a ref plus `take=`), which is curation's proposed `Sessions.ask(take=)` |
| "the one model call is a constrained verdict per pair" | `@judge.task def prefer(a: ArmResult, b: ArmResult) -> Verdict`: a typed, validated return |
| passes are fresh delegates whose memory is the tree | a task's fresh transcript is the same idea as `inherit="fresh"` |

**Trial-safe forks get a home.** curation.md calls trials "the most
dangerous thing in this document", because a fork inherits live host
objects, mounts and network, and it leaves the trial configuration
open. Under "whoever opens the world sets its environment", the gate
opens each arm, so it opens it with a safe `Profile`:

```python
profile.replace(host={"db": ReadOnlyDb()}, network=False)
```

That is substitution by the opener, which the rule already allows.
Agent-side narrowing on `ask` can wait. There is prior art:
`test_app(..., bind={"db": "testdb"})` (0.8.11) already runs handlers
against a substitute host object.

**`provide` / `host` is the library / service split.** It gives a
curator a rule for which to build:
- **A library** is code vendored into the world, running in the
  kernel with the session's own grants. Build one when the logic is
  computation the session should own and may patch (a patch is a bug
  report). It runs the same on dud, because world files travel into
  the guest.
- **A virtual backend service** is an `app/api/` app the builder
  writes and the gate publishes as a version. To a consuming session it
  is a **host** capability: it runs outside that session's kernel,
  under the service's own environment. Build one when the logic needs
  authority sessions shouldn't hold (credentials, a db, shared state).
  Sessions get a narrow surface instead of raw access, which is least
  privilege by construction.

**Services get their skills generated.** `scratch/reuse.md` derives a
service's contract by AST from its handlers, with docstrings as the
prose. That is the host-surface index mechanism above. A published
service's skill is that generated contract plus the builder's prose
about when to use it.

**One trace renderer serves the stack.** curation.md already copies
agex's chapter layout (`/chapters/<slug>/events/NNN-<type>.md`) on
purpose. With the neutral conversation plane and event schema from
plan step 1, one `TraceSource` renders agno and agex sessions alike.
curation's open question "a trace of your own session?" is agex's
`/chapters`, which the redesign keeps.

**Mounted vs vendored skills.** curation.md rejects live mounts for
skills: they "change under the agent and leave every trace ambiguous
about which code it ran against." The redesign mounts skills fixed by
the environment. Both hold, for different kinds of skill:
- **Skills that evolve** (curated, patchable, on probation) are world
  files, vendored at a version.
- **Skills fixed by the environment** (a package's docs, a service's
  generated contract) may be mounted, *because* each turn commit
  stamps an profile fingerprint (plan step 1).

The fingerprint is also where probation's cohort assignment for
services is recorded. Libraries record theirs in `.installed`.

**What curation adds to the harness contract** (plan step 1):
- run records with metrics, which is what `RunInfo` reads;
- the turn-commit stamps, including the profile fingerprint;
- harness state in reserved planes. curation's core rule is "no
  agent's tree has a second author", so the `task` outcome key must
  never be a file path.

**Gaps on the service path.** These are reuse.md's open phases, not
new problems:
- the peer resolver;
- versioned addresses, so probation can pin v4 vs v5 per session;
- the token-leak decision.

Services also widen what trials must fake: an arm calling a service
that writes shared state needs the resolver pointed at a fake or
read-only version.

**TS.** Prose skills follow the Agent Skills spec and could flow into
agex-studio. Code skills and services are Python and server-side, so
they can't. A `kernel:` key in skill metadata would let each side skip
what it can't run.

## Plan

1. **Carve the loop seams in nontainer and nontainer-studio, with agno
   as the only implementation and no behavior change.** This follows
   the dud ladder: carve the Executor seam first, add `DudExecutor`
   second. There are two contracts:
   - **nontainer harness contract.** It covers:
     - turn begin and end;
     - the point where inbox messages are delivered;
     - the conversation plane, named `__agno__/` today. It gets a
       neutral name;
     - commit boundaries;
     - the `SessionRunner` seam, including the loop that waits on
       delegates, and its async edges (see "Async at the Sessions
       edges" below);
     - **harness host objects**: the loop attaches its own objects
       (the `task` outcome object) to each session it runs. nontainer
       already does this internally for the `ws-*` verbs on dud.
       Deferred (see the decisions log, 2026-10-07): task forks get
       theirs when they open. The `agex` task object is not one of
       them: the embedder grants it through the profile ("Agent-defined
       tasks");
     - **run records**: each run's status, tool calls, tool errors,
       tokens and duration, persisted in the neutral run record. This
       is what curation's `RunInfo` reads, from agno's `RunMetrics`
       today;
     - **turn-commit stamps**: the `"runs"` stamp that agno's path
       writes since 0.8.6, plus an **profile fingerprint** (package
       versions, mounted skill versions, pinned service versions);
     - **reserved planes for harness state**: the `task` outcome key
       and anything else the loop persists live outside the paths
       ws-git counts as files, so they never read as the agent's
       uncommitted work.

     All of this lives implicitly in `adapters/agno.py` today. It
     moves into core as harness hooks. The spec is
     `~/git/nontainer/scratch/harness.md`: the tiers, the rules, the
     `turn` API, the conversation index, compaction's records, and the
     studio's `TurnDriver`.
   - **studio turn driver.** It covers:
     - building an agent for a session;
     - turning a run into SSE events;
     - cancel and resume-in-place;
     - retry and compression.

     This is spread across `turns.py`, `_build_agent`, `providers.py`
     and `compression.py` today.
   - **What agex needs from nontainer.** agex depends on nontainer
     (see the decisions log). Nothing below is agex-specific, so agno,
     MCP and server-side deployments benefit too:
     - **A neutral tool layer.** Each tool is defined once: name,
       description, parameter schema, sync and async call.
       `adapters/agno.py` and `adapters/mcp.py` each redefine every
       tool today and have drifted (MCP has no inbox and no ui note).
       agno, MCP and agex each wrap the one set, and agex never
       imports the agno adapter.
     - **A first-class ephemeral store** that writes nothing to disk.
       `workspace()` / `store()` default to `~/.nontainer`. With
       `kv=Memory()` the data stays in memory, but `path` still
       defaults there and holds the publication registry. A plain
       `grade(rs)` call needs a scratch world.
     - **`nontainer.Profile`**, the open-time config as one value.
     - **A harness-contract conformance kit**, shipped the way
       termish/monkeyfs ship `check_filesystem`. nontainer can't
       depend on agex, so agex and the agno adapter each run the kit
       in their own CI.

   The rest of the capability work (`provide` semantics,
   `HostObjectGrant`, scopes, the host index, `help()` metadata) is
   independent nontainer work and can land any time.
2. **Rewrite agex as a loop over nontainer** that supports both
   interaction styles, and plug it into the studio as a second loop.
   agno stays the default. Studio conformance tests run the same turn
   scenarios against both loops. nontainer gains three things in
   this step:
   - the value encoding from "Task values", not pickle. nontainer #110
     as filed is narrower (the return path of app handlers only), and
     it becomes a special case;
   - a lifecycle for ephemeral branches, for functional-style calls;
   - what agent-defined tasks need: waiting host calls off the clock,
     clean context for host-side async work, and the spec export (see
     "Agent-defined tasks").
3. **Shape freeze.** Write a versioned shape doc and pin two corpora,
   each with its generated JSON Schemas: the shape corpus (agex's API,
   in agex) and the harness corpus (nontainer's contract and
   `TurnEvent`, in nontainer). Their format is in
   `~/git/nontainer/scratch/harness.md`, "Loop corpus". Each Python
   signature in the shape doc sits next to a TS one, so Python-only
   idioms get caught while they are still cheap to change.
4. **The TS side catches up to the frozen shape.**
   - Rewrite agex-ts.
   - Add a TS world package: nontainer's semantics, limited to what a
     serverless studio needs, so no dud, Postgres or `serve.py`.
   - agex-studio gets the new kernel as a third `KernelAdapter`.
     `kernel-registry.js` already runs `py` and `ts` side by side, so
     existing sessions stay on the old `ts` kernel. The shell (gallery,
     gist publish, BYOK, viewer mode) is not rewritten.

### Async at the Sessions edges (step 1)

nontainer's core stays synchronous. Async is added only at the edges
where nontainer **waits**, **calls embedder code**, or **takes the
parent's lock**. Most of the `Sessions` API is quick bookkeeping that
is fine to call synchronously from anywhere:
- `ask` without `wait` returns a `Job` immediately.
- `list`, `result`, `take`, `outstanding`, `keep` and `cancel` take a
  short internal lock.

None of them needs an async twin. The studio hand-wrote glue for the
edges, which shows where nontainer should provide it instead:

- **Waiting.** Today `Sessions.wait()` blocks on a
  `threading.Condition`, and `ask(wait=True)` blocks until the answer
  arrives.
  - **Change:** add an awaitable answer stream, e.g.
    `async for name, answer in sessions.answers()`, plus an awaitable
    `wait`.
  - **It replaces:**
    - the studio's `_wake_on_answers` (`turns.py:393`), which turns
      `on_answer` into an asyncio queue via `call_soon_threadsafe`;
    - `StudioRunner._await_own_answers`, which polls
      `wait(timeout=slice)`.
- **The runner.** Today `SessionRunner.run` is sync and runs on the
  helper's thread pool.
  - **What the studio does today:** `StudioRunner` builds a fresh
    event loop for each delegate turn on that pool thread
    (`delegates.py` ~460), and it registers loop/task handles so
    shutdown can cancel across threads (`hold_delegate_run` /
    `stop_delegate_runs`).
  - **The cost:** each delegate holds a thread for its whole run,
    including model latency, so `max_workers=4` caps how many
    delegates run at once.
  - **Change:** accept a runner whose `run` is `async def`, detected
    the way `forked_at` support already is. Sessions then schedules
    those delegates as tasks on the embedder's loop. `max_workers`
    becomes a semaphore, and cancelling is `task.cancel()`. Sync
    runners keep the pool path.
  - The agex loop will almost certainly be async-native, so it needs
    this.
- **The parent's lock.** Today `ask` forks under the parent's
  `ws.lock`, and landing an answer reads the diff under the same lock
  (`_changed` / `_uncommitted`, `sessions.py` ~1161, ~1177).
  - **The problem:** once delegates are tasks on the embedder's loop,
    landing runs *on the loop*. A parent in the middle of `run_python`
    holds the lock for up to the sandbox timeout, so the whole server
    would stall.
  - **Change:** read the landing diff without the parent's lock. This
    is the same fix "Delegation from code" needs. Otherwise, landing
    has to push its sync part onto a thread.
  - A host-side `ask` called from a loop needs the same care: either an
    `aask` that forks off the loop, or a documented warning that the
    fork may wait on the lock.
- **Unaffected:** the agent-facing `sessions` tool, which runs on tool
  threads.

### Delegation from code (step 2)

Replaced by "Agent-defined tasks" (2026-10-08). Side-effect delegation
stays the `sessions` tool, and code gets tasks rather than `ask`.

- **The deadlock is fixed.** Landing an answer took the parent's
  `ws.lock`, which a `run_python` waiting on a delegate holds; landing
  now reads committed history through the child's handle (A5a).
- **Still open.** `ws.fork` in the middle of a call commits the call's
  partial writes under `{"tool": "fork"}`, splitting the `run_python`
  commit in two. A task's helper avoids it: an empty view forks from
  the last commit.

## Async agent code

This has its own doc: `~/git/nontainer/scratch/async.md`. It stands on
its own and doesn't wait for the redesign. In short:
- `run_python` accepts top-level `await` the way IPython does.
- The script runs on a private loop, never the host's.
- Async host objects work through nontainer's transport.

Issues so far:
- sandtrap #55 (`aexec` gaps), open;
- nontainer #176 (MCP sync tools blocked the event loop), fixed in
  0.8.12;
- nontainer #177 (async host methods refused at open), fixed in
  0.8.12.

Two pieces stay here because they belong to the loop seam: the async
edges of `Sessions` (step 1) and delegation from code (step 2).

## Parked

- **nontainer browser roadmap Phase 2′** (the harness in the tab, with
  the server as a kvgit remote). The browser belongs to TS, and
  agex-studio already is that far end.
- **A Remote `KernelAdapter`**, letting agex-studio drive a
  nontainer-studio server. This is a personal nicety, not the strategy.
- **World-format interop between Python and TS** (kvgit v1 vs v4,
  VFS layouts, the WireCommit remote). Not needed until a session has
  to cross languages.

## Risks

- **The gap between steps 2 and 4.** Old agex went through four
  breaking rebuilds in about four months. If the Python shapes are
  still moving when TS starts, TS ports a moving target. The shape
  freeze in step 3 exists to prevent that.
- **Two loops double the studio's turn-edge-case matrix**: cancel,
  resume, retries and compression. The cross-loop conformance tests
  are how that stays honest.

## Open questions

- Do server-side deployments stay on agno? That decides how far
  agex-next is pushed as a drop-in there.
- The harness contract is drafted in
  `~/git/nontainer/scratch/harness.md`. Its own open questions are
  listed there.
- Is it acceptable that a fork from inside `run_python` splits that
  call's commit in two?
- Where do host-object scope grants live: a reserved plane in the
  world (so a rewind would revoke them), or the embedder?
- Is agent-side narrowing on an agent-defined task (`hosts=[...]`,
  subset only) needed in v1? Curation's trial-safe forks don't need
  it: they use substitution by the opener (`profile.replace(...)`),
  which the environment rule already allows.
- Should skill metadata carry `kernel:` (python, ts) so each side
  skips skills it can't run?
- Should tasks declare `requires=[...]`, so they refuse to run in a
  world that lacks a capability?
- How do app handlers call tasks? Handlers run on a frozen snapshot
  with host objects from whoever serves, so there is no session,
  parent or budget. It needs:
  - an explicit grant per published app, since every visitor spends
    the server's key, plus budgets or rate limits;
  - a job shape (start, then poll or stream), since a task takes
    seconds to minutes and a handler is request/response;
  - a stand-in model for `test_app`, through `bind=`;
  - in agex-studio, the TS loop running in the tab with the device's
    own key, because there is no server.
- On dud, should a terminal host-side error be marked so the guest
  can't swallow the stop signal? (The recorded value already wins.)

## Decisions log

- 2026-10-03: agex-studio stays serverless and BYOK.
- 2026-10-03: Python goes first. agex is rewritten as a loop over
  nontainer that supports both interaction styles. nontainer-studio
  can run either agno or agex. agex-ts and agex-studio catch up
  afterwards, to the frozen shape.
- 2026-10-06: agex depends on nontainer.
  - agno's optional adapter lives in nontainer only because agno is
    third-party, so its glue has to sit on the side we control. The
    invariant nontainer enforces is that it imports nothing above it
    (`tests/test_layering.py`), and `agex → nontainer` keeps it.
  - Making agex standalone would mean re-growing a world protocol and
    in-memory backends, the very layer this redesign removes.
  - Putting agex inside nontainer would break "nontainer owns no model
    and no loop".
- 2026-10-06: Capabilities belong to the world, not the agent.
  - The registration API is nontainer's environment configuration:
    `provide` (code in the kernel) and `host` (authority on the host).
  - An agent carries a default environment only for the worlds it
    creates.
  - This resolves the earlier question of who owns `PythonConfig`:
    delegates inherit the parent's environment, whichever agent runs
    them.
- 2026-10-06: Documentation is skills. `visibility` and `describe` are
  dropped. A generated host-surface index and `help()` metadata cover
  what the model must know about host objects.
- 2026-10-06: pydantic-ai is the provider transport.
  - What: `pydantic-ai-slim` with per-provider extras, called through
    `pydantic_ai.direct` (`model_request` / `model_request_stream`).
    Its `Agent` is not used.
  - Two seams of agex's own:
    - a small provider protocol, roughly `stream(turns, tools,
      settings) → events, final turn, usage`, with pydantic-ai as its
      first implementation. A provider whose newest reasoning feature
      lags can get a direct-SDK adapter;
    - agex's own run record and event schema, mapped to and from
      pydantic-ai messages at the edge. A third-party message type
      never becomes the stored format (the `__agno__/` lesson), and the
      record is what TS mirrors at the shape freeze.
  - Pin a minor range; 2.54 releases fast.
  - Checked on pydantic-ai-slim 2.54.0 (type level only; no live
    provider calls):
    - `ThinkingPart` keeps `signature` and `provider_details`;
    - `ToolCallPart` keeps `tool_call_id`;
    - `RequestUsage` carries cache read and cache write tokens and
      cost;
    - there are Anthropic cache settings;
    - the provider list covers Anthropic, OpenAI, Google, Bedrock,
      OpenRouter, Ollama and others, plus a fallback model;
    - a scripted `FunctionModel` drove a tool call, a tool result and
      the final text through the direct API, and the history
      round-tripped through JSON.
  - Alternatives set aside:
    - LiteLLM: OpenAI-chat shaped, so reasoning round trips are
      squeezed; at most a proxy behind the OpenAI-compatible provider;
    - direct SDKs only: the most fidelity but three adapters, so kept
      as the per-provider fallback;
    - agno's model layer: it would re-couple to agno.
- 2026-10-06: Harness contract decisions.
  - Its spec lives in nontainer: `nontainer/scratch/harness.md`, with
    the studio's `TurnDriver` as a section.
  - The conversation plane becomes harness-agnostic:
    `__conversation__/`.
    - Core owns a small index: harness tag, session id, ordered run
      ids, forked-from.
    - Run bodies stay in each harness's own format.
    - Existing `__agno__/` sessions are read as a fallback and
      migrated on their next write, with the legacy keys removed in
      that same commit.
  - Both commit modes stay. agex defaults to per-call commits plus a
    trailing run commit, as the studio runs today.
  - The contract's shape is a context manager
    (`with ws.turn(run_id) as turn:`) over explicit `begin()` /
    `end()`, with an `async with ws.aturn(...)` twin.
- 2026-10-06: Loop corpus.
  - Two corpora in one format:
    - the harness corpus (tiers 1-5) in nontainer, run by the agno
      adapter, agex and later TS;
    - the shape corpus (agex's API) in agex, run by agex and agex-ts.
  - The format, loader and runner live in nontainer, and agex extends
    them.
  - Scenarios are written as Python dataclasses with builders. The
    per-scenario JSON is generated and committed, with a drift test,
    and TS reads it.
  - JSON over YAML: no dependency on either side, and readability
    matters less for a generated file.
  - The scripted model is the clock: outside events fire when it is
    asked for its next response.
  - Details: `~/git/nontainer/scratch/harness.md`, "Loop corpus".
- 2026-10-06: The rebuilt agex ships as 0.13.0: a breaking 0.x minor,
  pre-releases during the rebuild, `0.13.0` at the shape freeze.
- 2026-10-06: Each shape is declared once, in the lowest package that
  needs it; everything above imports it. The dependency chain is linear
  (studio → agex → nontainer), so there is no separate shapes package.
  - nontainer declares the turn statuses, the turn stamp, the
    conversation index, the corpus schema and **`TurnEvent`**.
  - agex declares its stored run record, `Outcome` and the task API.
  - The studio declares its SSE events and `TurnDriver`.
  - This refines the provider entry above: agex owns its *stored*
    record. The *streamed* events are nontainer's `TurnEvent`, because
    the harness corpus asserts them against more than one harness.
  - For TS, the dataclasses export JSON Schema, drift-tested beside the
    corpus JSON and pinned at the shape freeze.
- 2026-10-06: The studio's `TurnDriver` seam is built in plan step B6,
  against both loops, not before agex exists. The event-sink fix and
  `call_id` / `is_error` land early with agno alone (A6).
- 2026-10-06: B2 brings up Anthropic, OpenAI and OpenRouter first, and
  Google follows. Chaptering waits for curation's trace projection,
  which is what makes `/chapters` browsable.
- 2026-10-06: The open-time config bundle is `nontainer.Profile`, not
  `Env`. In nontainer, `env` already means a session's shell environment
  variables (`ws.runtime.env`). The profile carries those variables
  too, as `variables`, applied when the session opens. Read one back
  with `Profile.of(ws)`. The turn stamp's fingerprint key is
  `"profile"`.
- 2026-10-07: Harness host objects (A3c) are deferred. Host objects
  stay fixed when a world opens. agex gives each task fork its `task`
  object through `Store.fork(..., profile=...)`, which already works.
  - Attaching after open would be additive per executor: in-process,
    the policy plus each call's namespace; under process isolation, the
    sandbox's RPC handlers, which needs an add-handler API in sandtrap;
    on dud, the hostcall allowlist taken at open, with the `ws-*` verbs
    as precedent.
  - Human-in-the-loop requests can go through scopes checked at call
    time (nontainer #143), which return "needs input", rather than attaching
    objects mid-session.
  - So nothing is boxed in: agex declares its harness objects in one
    place, reads host objects per turn, and nobody mutates
    `host_objects` after open (nontainer #204 makes it read-only).
- 2026-10-07: The task calls in "Task calls": `task.success` /
  `task.fail` / `task.needs_input`, arguments bound by name, the
  `success` and `needs_input` statuses, the nudge, and the world
  defaults.
- 2026-10-07: Task values follow "Task values".
  - The declared type is the contract. It picks the encoding by kind,
    and decoding only ever builds that type.
  - In-process carries anything. Every other path carries exactly the
    encodable kinds, process isolation included, so moving a world
    from process to dud is never a breaking change.
  - The check runs when a task is called, against the world it runs
    in, never mid-run.
  - Inputs are values on every rung; a live input is a capability,
    bound as a host object of the task's fork.
  - A JSON schema exists only for data, and matters only where a
    value leaves Python: agent-defined tasks, apps and TS.
  - The encoding lives in nontainer, generalizing the handler-returns
    encoder; dud is unchanged.
- 2026-10-07: Agent code gets delegation through a harness host object
  (`ask`), never the embedder's `Agent`.
- 2026-10-08: The shape corpus (`agex.conformance`).
  - A task is written as JSON Schemas. `x-kind` marks what JSON has no
    type for: tables, arrays, bytes, and live classes, whose methods
    give canned replies and record their calls.
  - The model's script is neutral: `task_success` (with a JSON value,
    an input, or what a live input's method returns), `task_fail` and
    `task_needs_input`. Each harness writes them in its own language,
    so no scenario is written twice.
  - Values are readable JSON, read by their declared types, not
    `nt-value/1`'s Arrow and `.npy` blobs.
  - A scenario says where it runs: anywhere, in the process, or off
    it. A harness runs it on each of its rungs that fits.
  - The runner owns the script, so what the model was asked for is
    counted the same way for every harness.
  - The agex harness writes the task as a Python module and imports
    it, so a worker process or a dud guest gets its classes as it
    would an embedder's.
- 2026-10-08: nontainer keeps the record of a fold; each loop keeps
  its own folding. This replaces moving the algorithm into core
  (`turn.context` over a neutral `Msg`).
  - nontainer: the `__compaction__/` plane and `Fold`, when a fold is
    in force, rewind and fork behaviour, summaries never stored in
    runs, the `Compacted` event and the person's view. The studio
    reads these whichever loop wrote them.
  - Each loop: when to fold, what to splice, how to summarise,
    folding within a run, and chaptering. agno's adapter keeps its
    logic; agex writes its own, folding within a task's run too.
  - A neutral algorithm would be designed around agno's limits just as
    agex goes further (folding within a run, named ranges), and would
    serve one loop if agno leaves the studio.
  - The helpers (summary texts, `reduce`, `chunks`, token estimates)
    stay in `nontainer.compaction` as a library.
- 2026-10-08: Agent code defines tasks, through the `agex` host
  object; code-level `ask(returns=, agent=)` is dropped. This replaces
  the 2026-10-07 entry on `ask`. See "Agent-defined tasks".
  - Side-effect delegation stays the `sessions` tool (B5a); code gets
    the functional shape, which also works inside apps and helpers.
  - `agex` is a stubbed host object an embedder grants, not a
    primitive. Agent code holds the stub; the host half holds the
    embedder's `Agent`. Any harness on nontainer can grant it.
  - nontainer builds host objects per world (`HostObject(factory=)`),
    so the host half knows which world calls it, and decides by it.
  - A helper is a delegate branch of its caller, asked through the
    calling world's `Sessions` with `paths=[]` (forked from the last
    commit, landing nothing) and a runner of its own (`ask(runner=)`).
    The typed value stays in agex: the runner holds it, and nontainer's
    `Answer` carries text. Nothing rolls up into the calling run, as
    for delegates. A scratch world is the embedder's opt-out.
  - One level deep by default: a helper's profile has no `agex`.
  - Apps use the same object: the preview's helpers are the session's
    delegates, `bind=` gives stand-ins, tasks belong in POST handlers,
    and a published app's helpers go where its factory puts them.
  - Types cross as shapes. The stub compiles the signature in the
    kernel and sends it as data; the host runs nothing the guest sent;
    the helper's world gets generated classes; the caller's stub
    decodes into its own. Script-defined classes work, and the
    module-only rule goes.
  - Values only, on every rung: the full encodable set, and no live
    object, in-process included. Live inputs stay an embedder's
    task's.
  - Host objects marked as waiting don't count against `run_python`'s
    timeout, in sandtrap and in dud's supervisor; opt-in per host
    object. (Superseded 2026-10-09: every host call is host time.)
- 2026-10-09: The timeout bounds the code, not the host it calls: every
  host object's calls are host time, with no opt-in (sandtrap 0.4.2,
  dud 0.4.2, nontainer #222). A stub's own methods stay on the clock.
- 2026-10-09: A world's `agex` reaches the world's helper through
  `Sessions.of(ws)`: a `Sessions` registers itself over its workspace
  while open (nontainer #228), so no harness hands it over.
