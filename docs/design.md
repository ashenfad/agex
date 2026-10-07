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
| ends when | the model stops calling tools | `task_success(value)` |

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

# inside agent code, either shape
job = ask("write tests for app/", returns=TestReport, paths=["app/"])
reports = wait([job, job2])
```

Each front door has a sync and an async form (`say` / `asay`, plus
sync or `async def` tasks). The loop underneath is async-native.

### Outcomes come back through a host object

`task_success`, `task_fail` and "needs input" (covering both
clarifying questions and permission requests) are methods on one
`task` host object. It is a *harness host object*: the loop attaches
it to each session it runs, through the small hook in plan step 1.
nontainer needs no knowledge of agex.

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
  slot (return schema, destination) before each run. The single-writer
  rule makes the swap safe. Delegates are their own sessions with
  their own objects.
- **The value depends on the rung.**
  - In-process: the live object (the actual DataFrame).
  - Process isolation: a pickled copy. That is the known trust seam.
  - dud: JSON, bytes or file refs only.

  The typed codec (nontainer #110) fixes the last two. It is the same
  dependency as `ask(returns=)`.
- **The value is also written to the world.** The handler writes the
  encoded value under a reserved key, through the workspace's
  re-entrant lock (host objects calling back into the workspace is a
  tested invariant). It lands in the same commit as the call, so a
  resume, or a parent reading a delegate's ref, can find it.
- **sandtrap cleanup.** sandtrap can drop its leftover agex-specific
  `TaskSuccess` / `TaskContinue` handling.

### Spawn folds into delegation

| what spawn gave you | replacement |
|---|---|
| memoryless clone | `ask(task, inherit="fresh", paths=[])`, an empty view of a fork |
| typed result | `ask(..., returns=T)` |
| fan-out (`submit` / `map`) | non-blocking `ask` plus `wait(jobs)`; sync, no `async` needed |
| `max_spawns` | `Sessions(max_workers=)` |
| clone events streamed but never stored | delegates are real sessions: stored, open-able, resumable |
| depth-1 limit | delegation chain and budget |

The old dual-decorator "agents as functions" folds in the same way.
`ask(..., agent=grader)` runs a different agent on a fork of the
caller's world. It inherits that world's environment, so a delegate
can never hold more than its parent. It also gets resume,
observability and merge-back for free.

There are two real losses:
- **Live in-process return values.** A spawn could hand back a closure
  or a live object. Delegates return values through the typed codec,
  and large artifacts travel as files in the fork.
- **Cheapness.** A spawn was just a thread. An `ask` is a fork plus a
  runner turn plus commits. Forks are O(1); the real cost is the
  ephemeral-branch GC already planned for step 2.

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
  That core is harness-neutral, and `docs/compaction.md` reserves
  chaptering for later:
  - a chapter is a fold record with `first` set, plus a name;
  - the agent's own model writes the summary, using the agex chapter
    prompt;
  - the originals stay readable as files, via curation's trace
    projection.

  agex's compaction adapter is one more adapter, "of the same size as
  agno's".
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
       (the `task` outcome object, code-level `ask`) to each session it
       runs. nontainer already does this internally for the `ws-*`
       verbs on dud;
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
     `turn` API, the conversation index, compaction in core, and the
     studio's `TurnDriver`. `nontainer.compaction` (0.8.8)
     is the template: a harness-neutral core plus a thin per-harness
     adapter.
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
   - a typed value on `Answer`, via `ask(returns=…)`. This needs a
     typed codec on the view-exec return path (nontainer #110), not
     pickle;
   - a lifecycle for ephemeral branches, for functional-style calls;
   - delegation from code (see below).
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

No host object exposes sessions to agent code today. Code-level `ask`
arrives as a harness host object.

- **Shape.** A non-blocking `ask` returns a job, and a sync
  `wait(jobs)` gives `gather`-style fan-out with or without `await`.
  An awaitable wrapper (`wrap_future` on the helper's futures, behind
  a public accessor) is sugar on top. `ask(..., agent=other)` runs a
  different agent on the fork, which inherits the caller's
  environment.
- **Deadlock to fix first.** Landing an answer runs on the delegate
  pool thread and takes the *parent's* `ws.lock`. Agent code that waits
  for an answer inside `run_python` already holds that lock, so it
  hangs until the Python timeout. The lock-free landing reads above
  fix this.
- **Open.** `ws.fork` in the middle of a call commits the call's
  partial writes under `{"tool": "fork"}`. That splits the
  `run_python` commit into two.

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
- Is agent-side narrowing on `ask` (`hosts=[...]`, subset only)
  needed in v1? Curation's trial-safe forks don't need it: they use
  substitution by the opener (`profile.replace(...)`), which the
  environment rule already allows.
- Should skill metadata carry `kernel:` (python, ts) so each side
  skips skills it can't run?
- Should tasks declare `requires=[...]`, so they refuse to run in a
  world that lacks a capability?
- Should agent code get inline typed stubs, as spawn had with
  `@spawn.task def gen_svg(...) -> Resource`? If so, add them later as
  sugar over `ask`, not in v1.
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
