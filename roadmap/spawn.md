# Spawn: ephemeral typed clones for in-agent fan-out

!!! success "Status: implemented (v1) on branch `feat/spawn`"

    Built per this design. Core: `agex/agent/spawn.py` (`Spawn`,
    `SpawnTaskWrapper`); injected in `agex/eval/bridge/namespace.py`;
    clone via `BaseAgent._get_spawn_clone` (`agex/agent/base.py`);
    `max_spawns` on `Agent`; per-emission pool torn down in
    `agex/eval/bridge/__init__.py`; conditional primer in
    `agex/agent/loop/mixin.py`. Tests: `tests/agex/agent/test_spawn.py`
    (full suite green). One integration wrinkle found and handled: agent
    code receives the spawn task wrapped in a sandtrap `StFunction`, so
    `submit`/`map` unwrap `._compiled` (`Spawn._resolve_task`). Browser/
    Pyodide concurrency remains out of scope (agex-ts's domain).

    The original design notes below are kept as the rationale of record.

## Summary

Give an agent a first-class way to **spawn an ephemeral, memoryless clone
of itself to fulfill a typed subtask** — generate sample data, render a
plot, draft an SVG, research a topic — and get a validated result back.
The clone is the parent's own policy with no long-term memory; it runs
exactly one task loop and is discarded.

The surface reuses agex's existing contract: a `@spawn.task`-decorated
function with a return type the clone must satisfy. Spawning therefore
costs the agent no new mental model — it is `@agent.task`, one level down,
callable from inside agent code.

```python
@spawn.task
def gen_svg(prompt: str) -> Resource:
    """Create an SVG resource for a 64x64 tile. Use ..."""
    pass

tile = gen_svg("a small castle with a blue flag")   # blocks, returns Resource
```

Fan-out (heterogeneous — differently-shaped tasks at once) uses a
`concurrent.futures`-shaped surface, not `asyncio`. `spawn` plays the
role of the executor:

```python
h1 = spawn.submit(gen_svg, "a castle")          # SpawnHandle[Resource]  (Future subclass)
h2 = spawn.submit(gen_research, "medieval forts")  # SpawnHandle[Report]
h3 = spawn.submit(gen_data, params)             # SpawnHandle[DataFrame]

svg, report, data = h1.result(), h2.result(), h3.result()   # the trained reflex
```

The homogeneous case mirrors `executor.map`:

```python
tiles = spawn.map(gen_svg, ["a castle", "a forest", "a river"])  # list[Resource]
```

## Why

- **The dogfood pattern is too heavy for the common case.** Today an
  agent that wants a sub-agent must have `Agent` registered as a class
  (`examples/dogfood.py`), then hand-write `Agent()` + `.module(...)` and
  *know* what to re-register. For "produce one typed result," that
  ceremony is the bulk of the work.
- **No clean in-agent fan-out.** Existing sub-agent calls are
  deliberately synchronous (`_wrap_sub_agent_task` →
  `_sync_task_func`, `agex/eval/bridge/policy.py:117-158`) and block the
  whole turn. There is no idiomatic way for an agent to launch several
  subtasks at once.
- **The typed-result-from-a-clone shape is broadly useful.** Sampling,
  plotting, drafting, researching — anywhere the parent wants a
  validated artifact produced by a fresh reasoning context.

## 1. `spawn` — a pre-built clone in the eval namespace

`spawn` is injected into the agent's `__main__` eval namespace as a
pre-built object whose policy is **a clone of the parent's**. The agent
does not construct it; it is there, like a builtin.

- **Capabilities = the parent's policy.** The clone can use exactly what
  the parent can. This is the security story for free: child ⊆ parent,
  trivially, because child *is* parent's policy. Per-task narrowing (a
  capability allowlist, cf. the scopes doc's `defineTask({fns:[...]})`)
  is a future refinement; v1 is a full clone.
- **No shared VFS / cache.** The clone does not see the parent's
  `helpers/` or working-memory cache (§5b). Data crosses in through the
  **typed prompt argument**, which can be a rich object (DataFrame,
  Pydantic model) — no serialization, it's in-process.
- **`spawn` is a reusable template, not one-shot.** Define many
  `@spawn.task`s on it; each *invocation* allocates fresh ephemeral
  state. The `spawn` object itself is stateless — **policy is the
  template, state is per-invocation** (§2).
- **`spawn` is stripped from the clone's namespace.** Clones are
  depth-1 leaf workers, not orchestrators — this prevents unbounded
  recursive spawning and bounds cost. (Simpler than a depth counter and
  matches intent.)

## 2. The contract: `@spawn.task`

`spawn.task` is a decorator method on the injected object — *not* a new
decorator the agent defines, so sandtrap's decorator handling applies
directly (it re-applies decorators as explicit calls,
`sandtrap/rewriter.py:553`; `@spawn.task` becomes `f = spawn.task(f)`).

The decorated function's **signature, docstring, and return annotation
are the contract**, identical to host-level `@agent.task`. The clone's
loop validates `task_success(...)` against the annotation and retries on
mismatch. The return-type name (`Resource`, etc.) must resolve in the
agent's namespace — i.e. it must be a registered/known type.

**This is proven, not speculative.** `agent.task(fn)` is pure at
construction — signature inspection, empty-body validation, building an
inputs-dataclass, constructing the wrapper (`agex/agent/task.py:294-371`,
`_create_task_wrapper`). No host FS, no host imports, no escaping
mutation; the only mutation is registering the inputs-dataclass onto
*that agent's own* policy (`task.py:855`), which for a clone touches only
the clone. **Dogfood already builds a `Task` in-sandbox** (architect
calls `.task(...)` on a sandbox-constructed `Agent`) with an end-to-end
test (`tests/agex/test_end_to_end_dogfood.py`). spawn is the same path,
simpler: the clone's policy is pre-built, so the agent only *defines*
tasks, never registers capabilities.

Two properties that fall out for free:

- **Return annotation resolves lazily** at `task_success` time
  (`__expected_return_type__` in state →
  `agex/eval/bridge/result.py:121-143`). `-> Resource` only needs
  `Resource` resolvable when the clone *validates*, not when the agent
  *decorates*.
- **Blocking face is automatic.** Even an `async def` spawn task routes
  through `_sync_task_func` for in-sandbox callers (`policy.py:132`), so
  the direct-call surface blocks without any extra work.

**Policy is the template, state is per-invocation.** The agent decorates
`@spawn.task` once (the inputs-dataclass registers on the shared clone
policy then, single-threaded in the agent's turn); `spawn.submit(...)`
only *executes* that built task on fresh ephemeral state. Concurrent
fan-out reads the policy, never mutates it — the shared clone-template +
fresh-state-per-call split is exactly agex's existing policy/state
separation.

## 3. The surface mirrors `concurrent.futures`

The naming is chosen for **minimum agent error surface**: `concurrent.futures`
is the most-trained blocking-concurrency API in the corpus, so an agent
reaches for `submit` / `.result()` / `map` and gets it right with almost
no teaching.

- **Direct call** → blocks, returns the typed result.
  `gen_svg("a castle") -> Resource` (every function works this way).
- **`spawn.submit(fn, *args)`** → returns a `SpawnHandle[R]` immediately
  — a **subclass of `concurrent.futures.Future`**, so `.result()`,
  `.done()`, `.exception()` all behave exactly as trained. Typed `R` via
  the generic subclass.
- **`spawn.map(fn, iterable)`** → `list[R]`, mirroring `executor.map`.

Everything is **blocking** from the agent's point of view — no
`async`/`await` in the agent's vocabulary, ever. The agent already
experiences every call as "blocks, returns a value"; that uniformity is
what keeps the feature teachable in two sentences and **portable across
how the host invoked the parent** (sync or async).

**No `gather`.** It's an asyncio false-friend that tempts `await
spawn.gather(...)`, which fails in a blocking model. `.result()` and
`map` carry zero await-temptation — one obvious way, no asyncio bleed.
Heterogeneous collection is just multiple `.result()` calls (the submits
already launched everything onto the pool).

## 4. Concurrency is thread-based, always

The decisive simplification: **spawn concurrency never touches the
parent's event loop.** Each `spawn.submit(...)` runs the clone's *sync*
loop in a thread pool and returns a `Future`-subclass handle.

- LLM loops are ~99% network wait; sync HTTP releases the GIL, so the
  threads genuinely overlap → real concurrency.
- It works identically whether the parent is sync or async, because
  spawn doesn't use the parent's loop. When an async parent blocks on a
  `.result()`, its loop simply idles while the clone *threads* run —
  fine, it has nothing else to do.
- This extends the wisdom already in `_sync_task_func` ("avoid
  event-loop conflicts") one step: don't bridge agent-initiated
  concurrency through the loop at all.

We considered cooperative-asyncio fan-out (the async loop genuinely
awaits I/O — `async_loop.py:422`, `acomplete` clients). It works but
forces either parent-mode awareness in agent code or a sync→async bridge
that deadlocks inside a running loop. Threads give the same concurrency
with zero mode coupling. The async LLM clients remain relevant to the
*host-facing* async API; they are simply not what powers in-agent
fan-out.

**Backpressure:** N concurrent clones are N simultaneous provider calls.
The thread pool is bounded by an **agent-level cap** — `Agent(...,
max_spawns=N)`, defaulting to a conservative **8** (the real constraint
is provider rate limits). Excess `submit`s queue. The pool is
**per-parent-task**: created and torn down per top-level invocation, so
the cap stays cleanly per-agent and there is no process-global state.

## 5. State: ephemeral, no kvgit

Clones run on **`type="ephemeral"`** state — fresh per invocation,
discarded after (`Agent` default; `agent/__init__.py:198`). The
underlying store is `Live` (`agex/state/live.py`), a plain in-memory
`MutableMapping` with no versioning and no commits.

- **What this keeps:** cross-*turn* state *within* the one loop. The
  clone writes `x = ...` in turn 1 and uses it in turn 3; its event log
  accumulates so the LLM sees its own history. `Live` handles this
  in-memory for the loop's lifetime.
- **What this drops (correctly):** cross-*task* memory, checkpoints,
  time-travel into the clone, `commit_hash` on clone events, crash
  recovery (re-spawn instead).
- **This dissolves concurrent-commit contention.** No commits → no
  branch to isolate → each clone is just a separate in-memory `Live`
  object, naturally isolated. Choosing ephemeral isn't only a
  simplification; it's what makes the threaded fan-out safe on the state
  axis.
- **The parent still records the I/O boundary.** On whatever backend the
  parent uses (possibly versioned), its action event captures "called
  `gen_svg`, observed this `Resource`." Replay-consistent: agex
  re-reasons over committed state and never re-executes prior emissions,
  so resuming the parent doesn't re-run the spawn — the observed result
  stands. (Same reason the scopes doc says clones are "re-spawned, never
  resumed.")

## 5b. VFS and cache — blank by default, and that's free

Both the VFS and the "cache" are just keys in the state store
(`__vfs_*` for files, `__cache__/*` for the persisted namespace —
`agex/cache.py`, `monkeyfs` virtual backend). So a clone on a fresh
`Live` store gets an **empty VFS and empty cache automatically** —
isolation is the default, not something we build. This matches existing
sub-agent behavior: a sub-agent already resolves its *own* state config
and gets an isolated cache (`test_cache.py:287-331`).

- **Cache: never inherit.** The cache is the parent's cross-turn
  *working memory* — variables it assigned mid-task. A clone that
  inherited it wouldn't be memoryless; it'd be a fork of the parent's
  mental state. Blank cache is both correct and the default.
- **VFS: blank in v1.** A *value* (DataFrame, your model) reaches a clone
  through the **typed prompt argument** — in-process, precise, no
  serialization. That covers the common generative case ("plot *this*
  data") without softening isolation.

### Read-only VFS mounting (future, must stay reachable)

The "settled" framing above is incomplete, and v1 must not foreclose the
fuller story. The guiding principle:

> **Match the channel to the shape of what you hand over.** A *value* →
> typed argument. A *corpus* (codebase, doc tree, many files) →
> read-only filesystem mount.

For FS-shaped subtasks — "explore and summarize the auth module," "search
these docs" — handing the corpus as a typed arg is wrong: it would
serialize thousands of files into one object and load them whole,
defeating agex's premise that an agent navigates *selectively* (`ls`,
`grep`, read file X). Those subtasks want a *filesystem*, read-only.

This is a **feasible future opt-in**, deferred but not blocked:

- **Mechanism exists.** `ReadOnlyFS` + `MountFS` (as used by
  `agex/fs/skills_vfs.py`, `chapters_vfs.py`): clone gets its own blank
  *writable* scratch layer plus a read-only mount of the parent's VFS
  (whole or a subtree). Reads everything, can't corrupt the parent tree.
- **No staleness.** The parent is *suspended* at the spawn call (blocked
  on the result / `gather`), so it isn't mutating files while clones run
  — every clone sees a consistent read-only view as of spawn time, even
  under fan-out.
- **No escalation.** Clone ⊆ parent already; anything in the parent's VFS
  the parent could read itself. So it's safe for the *agent* to choose
  per-task (a host-side mount allow-list is a later refinement).
- **Subsumes helper-sharing.** A clone with read-only VFS access also
  sees the parent's `helpers/` and can `import` them — partly resolving
  the helper-promotion thread for free.

**Forward-compatibility guardrails for v1 (do these now so the road stays
open):**

1. **`spawn.task` is a decorator *factory*, callable bare or with
   kwargs** — `@spawn.task` *and* `@spawn.task(fs=...)` both work from
   day one, even though `fs=` is rejected/unused in v1. Locking it to a
   bare decorator would force a breaking signature change later.
2. **Build the clone's FS through the mount-capable assembly path**
   (`prepare_task_loop` already wraps in `MountFS` for chapters/skills,
   `state_helpers.py:206-215`) — not a hardwired `VirtualFS(Live())`.
   Adding a read-only overlay later must be "add a mount," not
   "restructure FS construction."
3. **Don't assume zero overlays anywhere** in the per-spawn state/FS
   allocation; keep the clone's FS pluggable.

Deferred (not v1): copy-on-write (clone edits a file, parent doesn't see
it — opens "do changes flow back?"); host-side mount allow-lists;
write-back. The likely v1-plus shape is a per-task `fs=` arg
(`None` blank / `True` whole-RO / `"/path"` subtree-RO), with mount
location (root-overlay vs. a `/workspace` subpath) the live open
question.

## 6. Observability — events forward, tokens opt-in

The plumbing exists: events carry `agent_name` + `full_namespace`
(`agex/agent/events.py:282-283`, stamped in
`agex/state/log.py:59-65`); `TokenChunk` carries the same
(`agex/llm/core.py:129-130,236`); sub-agent calls already forward
`on_event`/`on_token` from contextvars (`policy.py:137-143`). Consumers
demux interleaved fan-out streams by `full_namespace`.

Decisions for the spawn case:

- **Events forward by default; tokens are opt-in.** Events are
  low-volume — auto-forward (opt-out to silence), matching existing
  sub-agent behavior. Token streams are the volume risk: a 20-way
  fan-out is 20 concurrent token firehoses. Most fan-out UIs want final
  results plus per-clone status, not 20 live streams. Make clone token
  streaming opt-in.
- **Label Live clones explicitly.** A root `Live` state sets
  `full_namespace = agent_name` (`log.py:64-65`), so a clone of the
  parent would be indistinguishable from it (and from sibling clones).
  Wrap each clone's `Live` in `Namespaced(Live(), "<unique tag>")` (e.g.
  `spawn:gen_svg:0`) purely for the label — `Namespaced` is key-prefixing
  over any `MutableMapping`, works over `Live`, needs no kvgit. Separate
  `Live` objects isolate storage; the `Namespaced` layer only labels the
  stream.
- **Do not route clone events into `_current_parent_log`** (the
  contextvar at `policy.py:37` that persists sub-agent events into the
  parent's log). For ephemeral clones we forward to `on_event` (stream)
  but skip parent-log persistence — else we reintroduce the commit
  contention we just removed. **Stream, don't store.**
- **Async-handler-from-thread bridge (real new plumbing).**
  `add_event_to_log` schedules async `on_event` via
  `asyncio.get_running_loop()` (`log.py:99`); in a worker thread there is
  no running loop, so it hits the `RuntimeError` branch and *closes the
  coroutine* (`log.py:101-104`) — async handlers would silently drop
  clone events. To support them, capture the parent's loop up front and
  `run_coroutine_threadsafe(...)` onto it. Plus `copy_context()` into
  each worker so the handler contextvars are visible at all. Sync
  handlers work as-is.

## 7. Scopes

Already specified by the scope-interrupt design's "spawned composition":
the clone inherits the parent's **grant snapshot** at spawn, cannot
self-grant, cannot durably suspend (it's ephemeral), and a scope-need
**surfaces up to the parent as a structured failure** carrying the scope.
`spawn` is the concrete realization of that section — no new mechanism.

## 8. Error semantics — inherited from `Future`, not designed

Because the handle *is* a `concurrent.futures.Future`, error handling is
whatever the agent already knows — no new flag, no new rule:

- **`h.result()` re-raises** that clone's failure (a sub-agent `TaskFail`
  → `EvalError`, as `_wrap_sub_agent_task` already converts). Familiar.
- **`h.exception()` inspects without raising** — the trained
  partial-success idiom ("7 of 8 tiles succeeded is usable": iterate
  handles, take `.result()` where `.exception()` is `None`).
- **`spawn.map` raises on the first failing element** when consumed,
  mirroring `executor.map`.

This is strictly less to teach than an asyncio-style `return_exceptions`
flag: mirroring the API gives both raise-and-inspect behaviors for free.

## 9. Setup / init inheritance

A clone **re-runs the agent's `init` / setup-namespace** per invocation.
Setup is part of the agent *definition* — closer to a constructor than to
working memory — so a clone of the agent is a freshly-initialized agent
(distinct from the cache, which is never inherited, §5b). Caveat: under
fan-out this runs `init` N times; if `init` is expensive, a future
optimization could cache the seeded namespace off the template (policy
*is* the template, §2). v1 keeps it simple and correct.

## Decisions

- **Surface / naming** → mirror `concurrent.futures`: `spawn` is the
  executor (`spawn.submit(fn, *args)` → `Future` subclass with
  `.result()`; `spawn.map(fn, iter)`); no `gather` (§3). Chosen for
  minimum agent error surface.
- **Concurrency model** → threads, always; never the parent's event
  loop (§4). Browser/Pyodide out of scope.
- **Concurrency cap** → agent-level `max_spawns`, default 8; per-parent-
  task pool (§4).
- **Error semantics** → inherited from `Future` — `.result()` raises,
  `.exception()` inspects; no `return_exceptions` flag (§8).
- **State** → `ephemeral` / `Live`, no kvgit (§5).
- **Setup/init** → clone re-runs the agent's `init` per invocation (§9).
- **Recursion** → strip `spawn` from clones; depth-1 leaf workers (§1).
- **Capabilities** → full clone of parent policy; per-task narrowing is
  future (§1).
- **VFS / cache** → blank, isolated by default (free); cache never
  inherited; read-only VFS mounting deferred but kept reachable via
  three v1 forward-compat guardrails (§5b).
- **Task-builder** → verified safe; proven by dogfood + test (§2).
- **Tokens** → opt-in forwarding; events forward by default (§6).
- **Parent-log persistence of clone events** → no; stream-only (§6).
- **Async handlers from threaded clones** → ship sync-handler support;
  the `run_coroutine_threadsafe` bridge is additive, deferred (§6).

## Open questions

None gating. Remaining items are deferred future work, not v1 blockers:

- **Read-only VFS mounting** shape — the `fs=` arg and root-overlay vs.
  `/workspace` subpath mount location (§5b). Future; v1 guardrails keep
  it reachable.
- **Per-task capability narrowing** — depends on a future per-task fn
  allowlist (§1).
- **`init`-seed caching** off the template, if per-invocation `init`
  proves expensive under fan-out (§9).

## Non-goals (v1)

- **Not persistent.** Clones have no cross-task memory; not resumable
  (re-spawn instead).
- **Not narrowed.** No per-task capability allowlist yet; full clone.
- **Not asyncio-exposed.** No `await`/`gather` in agent code; concurrency
  is the blocking `concurrent.futures` surface. Browser/Pyodide
  concurrency is out of scope (agex-ts's domain).
- **Not recursive.** Clones can't spawn (depth-1).
- **Not an orchestration DSL.** It's plain blocking calls + handles.

## Implementation sketch (change-list)

- **`spawn` injection** — build a policy-clone of the parent and inject
  it into the `__main__` eval namespace at sandbox-prep time
  (`agex/eval/bridge/`), with `spawn` itself excluded from the clone's
  namespace.
- **`SpawnTask` / `spawn.task`** — a runtime task-builder usable from the
  sandbox: signature/docstring/annotation → contract → callable, as a
  decorator *factory* (bare or kwarg'd, per §5b guardrail #1). Direct
  call runs the clone's sync loop on ephemeral state and returns `R`.
- **`SpawnHandle[R]`** — a generic subclass of
  `concurrent.futures.Future` (for typed `.result()`).
- **`spawn.submit(fn, *args)`** — submit the clone's sync loop to the
  bounded per-task pool; return a `SpawnHandle[R]`. `copy_context()` into
  the worker.
- **`spawn.map(fn, iterable)`** — homogeneous sugar, mirrors
  `executor.map`.
- **State** — per-invocation `Namespaced(Live(), unique_tag)`; no kvgit
  path; build the clone FS through the mount-capable path (§5b guardrail
  #2).
- **Observability** — forward `on_event` by default (namespaced), gate
  `on_token` behind opt-in; sync handlers supported now, the
  async-handler-from-thread bridge (`run_coroutine_threadsafe` onto the
  captured parent loop) deferred.
- **Primer** — one short, mode-free section: define a `@spawn.task` for a
  typed subtask and call it; `spawn.submit` + `.result()` (or
  `spawn.map`) to run several concurrently; a one-line note that spawn
  tasks cost a full loop.
