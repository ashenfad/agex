# agex rebuild: implementation plan

Status: in progress (2026-10-08).
- **Phase A** is merged through delegation: A0-A3 shipped in nontainer 0.9.0/0.9.1, and A4, A5 and A8a-b are on nontainer `main`. The next release, 0.10.0, closes A5 and ships the kit (A7); it is agex's floor. The studio's A5 PR (nontainer-studio #90) is a draft against unreleased `main` until then.
- **Phase B** is under way on `rebuild`: B0-B4 and B5a (the `sessions` tool) are merged, and B5b (agent-defined tasks) is next.
- **Still open in Phase A:** A6 (studio fixes), A8c (large values spill to the plane) and A3c (deferred). The async agent code track must land before the shape freeze (see Parallel tracks).

This plan covers *when* and *in what order*. The *what* and *why* live
in:
- `docs/design.md`: the shape, the decisions log, what
  stays and what goes;
- `~/git/nontainer/scratch/harness.md`: nontainer's harness contract,
  the studio's `TurnDriver`, the loop corpus;
- `~/git/nontainer/scratch/async.md`: async agent code.

## Ground rules

- **Phase A is nontainer and the studio, and every step lands with no
  behavior change.** The full nontainer and studio suites pass
  unchanged unless the step names a fix. Production deployments run on
  nontainer, so they must stay safe at every step.
- **One step is one PR,** or a short stack. The user drives merges and
  releases. A nontainer release closes each milestone, and the studio
  raises its floor after it.
- **Every step's exit names the corpus scenarios that pin it,** once
  the corpus exists (A3 onward).
- **The rest of the stack stays as it is:**
  - Python >= 3.10, which nontainer and pydantic-ai already require;
  - tests through `uv run pytest`;
  - Conventional Commits.
- **agex is a clean break, released as 0.13.0.** That is a breaking
  minor, which 0.x allows, and the user is its only real user. Nothing
  is published during Phase B: it stays `0.13.0.dev0`, and the studio
  uses it from a local editable install. B6's tests, which run the
  studio over both loops in CI, take agex as a git dependency.
  `0.13.0` goes out at the shape freeze. The CHANGELOG says plainly
  that it is a new API.
- **Phase B lands on `rebuild`, not `main`.** Each step is a PR into
  `rebuild` (CI runs there too), so `main` stays the working 0.12 code.
  At the shape freeze, `rebuild` merges into `main` with a merge commit,
  keeping each step's history, and `0.13.0` is released from it.

## The stages

```
A0 small bits ─┐
A1 tool layer ─┼─► A3 turn API + TurnEvent + corpus ─► B0 skeleton ─► B1 sessions ─► B2 providers ─► B3 tasks
A2 conv. index ┘          │                                                                         │
                          ├─► A4 compaction contract ─────────────────────────► B4 compaction ◄─────┤
                          ├─► A5 delegation ──────────────────────────────────► B5 delegation ◄─────┤
                          └─► A7 kit ships (with A4/A5's scenarios)                                 │
A6 early studio fixes (agno only; any time)                                                         │
                                       B6 studio seam, both loops (needs B1-B3) ◄───────────────────┘
                                                              ─► C freeze ─► D TS
```

Only A0-A3 come before agex starts. A4, A5 and A7 run alongside
B1-B3. A8, the value encoding, lands between B3a and B3b. The studio's
`TurnDriver` seam is built in B6, against both loops, rather than in
Phase A.

## Phase A: nontainer and the studio

Each step follows `harness.md`'s build order.

### A0. Small prerequisites (before B0)

**Status: done.**
- nontainer #196: the memory store, `Profile` (first built as `Env`)
  and the agno instruction fix. Shipped in 0.9.1.
- nontainer-studio #88: the stale comment. Merged 2026-10-07.
- Built differently from what follows below:
  - the bundle is `Profile`, not `Env`, because in nontainer `env`
    already means a session's shell environment variables
    (`ws.runtime.env`);
  - `Profile` carries those variables too, as `variables`;
  - the read-back is `Profile.of(ws)`, not a `ws` attribute.

- **A first-class ephemeral store**: nothing on disk, registry
  included. A plain `grade(rs)` needs a scratch world.
- **`nontainer.Profile`**: today's open-time config as one frozen value
  (python grants, host objects, mounts, commands, skills). It is
  accepted by `Store.open`, and forks inherit it. It only bundles what
  exists now. The new capability semantics (`provide`,
  `HostObjectGrant`, scopes) come later, in the capabilities track.
- **Fixes found while extracting the contract:**
  - the turn-mode instruction line (agno.py:790-792);
  - the stale `_resume_once` comment (studio sessions.py:968).

  Dropped tool errors are fixed in A6, with `is_error`.

**Exit.** A store opened with no path writes nothing to disk. `Profile`
round-trips through `Store.open` and a fork. Suites pass.

### A1. The neutral tool layer

**Status: nontainer #198 merged 2026-10-07 (branch `refactor/a1-tool-layer`).**
- It lives at `nontainer.adapters.tools`, not `nontainer.tools`: it
  uses the descriptions and the sessions and apps layers, which the
  layering test keeps core from importing.
- The description surface is byte-identical, pinned by a golden file
  recorded from `main` first.
- `PYTHON_UI_NOTE` (the `ui=` teaching, which promises display beside
  the reply) stays the agno toolkit's to append. Only a rendering host
  can keep that promise.
- MCP's `run_python` now reports saved artifacts. That is the named
  fix.

Each tool is defined once in `nontainer.adapters.tools`: name, description,
schema, sync and async call, and a result carrying text, media and
`is_error`. The instructions and skills catalog move into the shared
render. agno's toolkit and the MCP server are rewrapped on the one
set.

**Exit.** Both adapters are built on `nontainer.adapters.tools`, and suites
pass. MCP's `run_python` gains the ui-artifacts note, a named fix.
Tool text is otherwise byte-identical: a snapshot test compares the
rendered descriptions before and after.

### A2. The conversation index and plane, with migration

- `__conversation__/index` (core-owned) plus `runs/<id>`
  (harness-owned).
- Read fallback from `__agno__/`, and migration on first write, with
  the legacy keys removed in the same commit.
- Core fork (fresh or full), delete and "inherited only" run over the
  index. `_rebind_conversation` stops writing agno's field names.
- `KvgitSessionDb` / `KvgitStoreDb` read and write through the index.
- The studio's four direct reads of agno storage move to core calls:
  `_inherited`, `db.delete_session`, `fork_session`, and `agent.db`
  in `_keep_aborted_run`.

**Exit:**
- A migration test on a fixture store written by 0.8.13, on both
  disk and Postgres backends: read before a write, migrate on the
  write, fork both ways, delete.
- **A dry run on a copy of a real studio store and a copy of a production
  store.** Production stores are the risk here.
- Suites pass.
- **Storage note:** once a session migrates, nontainer releases older
  than this one can't read its conversation. That needs a CHANGELOG
  warning and a minor version bump.

**Status: done.** nontainer #199, shipped in 0.9.0.
- **How it shipped:**
  - **The plane.** Three keys: `__conversation__/index`, `/record` (the
    harness's own session record, opaque to core) and `/runs/<id>`. Read
    and written through `nontainer.conversation`.
  - **No migrate API, no dry-run mode.** Migration is lazy: a head with
    no index is read from `__agno__/`. The fallback is semi-permanent.
  - **The agno db** refuses to write over another harness's
    conversation. Both prefixes take `ours` in a merge.
- **Exit, as met:**
  - **Migration tests** (`tests/test_conversation.py`) seed the 0.8.x
    `__agno__/` layout directly rather than using a stored 0.8.13
    fixture. CI's Postgres job runs them on Postgres too.
  - **Studio-store copy:** done. All 10 sessions read, migrated, forked
    both ways and deleted cleanly.
  - **Production-store copy:** the user runs it, with the read-only
    `verify_conversations.py` in #199's description.
- **The studio's four call sites** moved to core calls after the
  release ("reach the conversation through nontainer's core", which
  requires 0.9.0).

### A3. The turn API, harness host objects, and the corpus machinery

- `ws.turn(run_id)` / `ws.aturn(...)` over `turns.begin` / `end`,
  covering:
  - how a turn ended (completed, cancelled, failed, interrupted);
  - resume in place;
  - the reentrancy refusal;
  - the turn stamp with `runs` and the `profile` fingerprint (built
    from `PythonConfig` and the mounts until `Profile` is everywhere);
  - **harness host objects**, attached for each session.
- The agno adapter is re-expressed on it:
  - `begin_turn`, `deliver` and `end_turn` plus the db's commit become
    the turn API;
  - `keep_aborted_run` and the cancel-path `settle` become the turn's
    ending.
- **`TurnEvent`**, the streamed vocabulary (`harness.md`, "Turn
  events"). The agno adapter's harness-under-test maps agno's
  `RunEvent`s onto it.
- **Corpus machinery**, in `nontainer/conformance/`:
  - the dataclasses, builders, loader, JSON export and drift test;
  - **JSON Schema export** of the corpus schema and `TurnEvent`,
    covered by the same drift test;
  - the runner, and the agno adapter's harness-under-test with its
    list-driven scripted agno `Model`.

**Exit:**
- Harness scenarios for tiers 0-3 pass on the agno adapter:
  - completed;
  - cancelled after a tool, with the closing note, the inbox settled
    and one run commit;
  - failed;
  - interrupted then resumed, keeping its tool results;
  - checkout rewinds the conversation;
  - a fork carries or drops it;
  - an inbox note delivered once, requeued after a discarded attempt,
    restored after a failed attach.
- The studio suites pass unchanged.
- **Milestone: a nontainer release** (A1-A3; a minor bump because of
  A2's storage change).

**Status: done.** A3a (#200) and A3b (#201) shipped in 0.9.1; A3c is
deferred (below).
- **A3a: nontainer #200.**
  - **What it adds:**
    - `nontainer.turns`: `TurnEvent` (`kind` = class name) and `RunStatus`.
    - `nontainer.conformance`: corpus format, runner, dependency-free codec and JSON Schema, export plus drift test.
    - 11 tier 0-3 scenarios.
    - `adapters/agno_conformance.AgnoHarness`, which runs the adapter as documented. It ends aborted runs the studio's way, and classifies errors by `RunError.error_type`.
  - **Capabilities and known gaps:** `resume` / `keeps-aborted-runs`, found by feature check (`continue_run` takes `continue_from`, agno 2.8.5 and later). Known gaps are strict: a gap that closes fails until removed.
  - **Reference harness:** in `tests/test_corpus.py`, on core's conversation plane. It passes every scenario with no gaps.
- **A3b's to-do, the gaps the corpus recorded:**
  1. A cancel lands two run commits: `keep_aborted_run` re-saves.
  2. A failure is stamped `error` and commits twice.
  3. An interruption is stamped `error`.
  4. agno 2.1 runs no post hook for a streamed run, so notes are never settled there.

  The turn API's ending (one commit, contract statuses, settle by construction) should close all four.
- **Deferred:** the agno-specific inbox requeue and restore scenarios (they need run-level retries).
- **A3b: nontainer #201.**
  - **What it adds:**
    - `ws.turn` / `ws.turns.begin` / `turn.end` (`nontainer.turns.Turn`), with one commit per turn stamped `{"tool": "turn", "runs": {id: status}}`.
    - `TurnInProgress`.
    - The agno db and `end_turn` stage under an open turn.
    - `adapters.agno.finish_turn` / `status_of_error`.
  - **Gaps:** `AgnoHarness` has no known gaps left.
  - **The studio:** its suite passes unchanged (it opens no turns yet).
- **Split out as A3c:** harness host objects (host objects are fixed at open via PythonConfig today) and the profile fingerprint in the turn stamp (manifest contents undecided). Neither blocks a loop that owns its turns; agex needs host objects by B3.
- **Milestone release:** 0.9.1, A1-A3b. A3c can follow in a later release.
- **A3c deferred (2026-10-07).** B3 doesn't need it: a task fork gets its `task` object through `Store.fork(..., profile=...)`, which already works. Why it waits, and what keeps it open, is in the redesign doc's decisions log (2026-10-07):
  - attaching after open is additive per executor (the in-process policy and namespace, sandtrap RPC handlers, dud's hostcall allowlist);
  - human-in-the-loop requests can use scopes checked at call time (nontainer #143) instead;
  - the guardrails: agex declares its harness objects in one place and reads host objects per turn, and `PythonConfig.host_objects` is read-only after open (nontainer #204).

  The profile fingerprint waits with it.

### A4. The compaction contract

**Status: done.** nontainer #209, merged 2026-10-08; ships in 0.10.0.
Six tier 4 scenarios pass on agno (2.5.0, 2.8.5, 3.0.1) and on the
reference harness, which folds in about fifty lines on the core's
records and helpers.

**nontainer keeps the record of a fold; each loop keeps its own
folding.** The algorithm does not move into core (decided 2026-10-08,
replacing the earlier `turn.context` plan).

What nontainer owns, because every harness and the studio read it:
- the `__compaction__/` plane and the `Fold` record, and the rule that
  a fold is in force only while its anchor is in the history;
- how folds behave in a world: a rewind takes them back, a fork
  carries them, a fresh fork drops them, and a summary never enters a
  stored run;
- the `Compacted` event, and the person's view of a fold (the
  studio's marker that opens to the summary).

What each loop owns: when to fold, what to splice into the next
request, how the summary is written, folding within a run, and
chaptering. agno's adapter keeps its logic as it is. The helpers
(the summary texts, `reduce`, `chunks`, `estimate_tokens`) stay in
`nontainer.compaction` as a library a loop may use.

The work:
- `docs/compaction.md` describes the record and its rules as the
  contract, and the folding as the agno adapter's;
- the harness corpus gets compaction scenarios, which check only what
  can be observed. Scripted model steps report token usage, so a
  scenario can cross a budget on demand.

**Exit.** Compaction scenarios pass on agno:
- a fold over budget;
- the fold in force in later requests;
- no summary in stored runs;
- a rewind takes back its folds;
- a fresh fork drops them.

The studio's compaction tests pass.

### A5. Delegation

**Status: nontainer side merged 2026-10-08; ships in 0.10.0. The
studio's half is nontainer-studio #90, a draft until that release.**
- **A5a, lock-free landing (#210).** Landing an answer reads committed
  history through the child's handle, never the parent's workspace
  lock, which a `run_python` waiting on the delegate holds. The
  deadlock regression test runs on every rung, each case in a
  subprocess.
- **A5b, async runners (#211).** A runner whose `run` is `async def`
  is scheduled on the embedder's loop (`Sessions(loop=)`), with
  `max_workers` as a semaphore, and `cancel` stops it; `aask`,
  `await_ready`, `answers()` and `aclose`.
- **A5c, waiting (#212) and the tier 5 corpus (#213).**
  `until_settled` / `auntil_settled` run a delegate's turns until
  neither its own delegates nor a note in its inbox is outstanding,
  within `max_wakes`; `Turn.opening()` / `aopening()` is a woken
  turn's first message. The corpus scripts delegates
  (`Scenario.delegates`, held until a `delegate_answers` event
  releases each), and a harness supplies them through
  `Harness.delegation()` (`adapters.corpus_delegates`), since the core
  runner may not import `sessions`.
- **Found by the studio PR (#214):** `until_settled` spun on a helper
  that had closed (`Sessions.closed` now ends the wait), and its unread
  note pointed the asker at a verb that cannot reach its delegate's
  jobs.
- **The studio (#90):** `StudioRunner` is an async runner on the
  server's loop, waiting through `auntil_settled`; shutdown cancels
  jobs through their helpers. The waker stays on `on_answer`:
  `answers()` collects, and would take answers from the toolkit's
  mid-turn delivery and the transcript's.

The plan as written:
- `sessions.until_settled(run_turn, max_wakes=)`;
- async `SessionRunner`s scheduled on the embedder's loop, with
  `max_workers` as a semaphore;
- an awaitable answer stream;
- lock-free landing reads;
- an `aask` that forks off the loop.

The studio's `_await_own_answers`, `_wake_on_answers` glue and
per-delegate event loops are replaced.

**Exit:**
- Tier 5 scenarios pass:
  - a runner commits before it returns;
  - the wait loop names unread delegates when its budget runs out;
  - an answer is delivered exactly once mid-turn.
- **A regression test for the deadlock** where code waits on a
  delegate inside `run_python`.
- The studio's delegation tests pass.

### A6. Early studio fixes (agno only; any time)

These don't need the seam:
- **Make the event sink safe across loops and threads.** Delegate
  turns emit from a fresh event loop on a worker thread into one
  `asyncio.Condition`. A5's async runners remove those loops
  (nontainer-studio #90 runs every delegate on the server's loop), but
  the sink should be safe regardless: a registry outside a server
  still runs delegates on a loop of its own.
- **`tool_start` and `tool_end` gain `call_id` and `is_error`.** The
  frontend pairs tool ends by id instead of by name, and agno's
  `ToolCallError` stops being dropped.

**Exit.** A test that emits from a worker-thread loop while a follower
waits on the server loop. The frontend pairs tool calls by id. Suites
pass.

The `TurnDriver` seam itself moved to B6.

### A7. The conformance kit ships

The harness corpus is completed for tiers 0-5 and shipped in the
package, the way `check_filesystem` is. It runs alongside B1-B3, once
A4's and A5's scenarios exist.

**Exit:** a nontainer release. That release is agex's floor.

**Status:** tiers 0-5 are in `nontainer.conformance` on `main`;
0.10.0 is the release.

### A8. The value encoding (between B3a and B3b)

**Status:**
- **A8a, the module (`nontainer.values`):** merged (nontainer #205).
- **A8b, the transport on each rung:** typed host data and classes
  (#206), stubbed host objects (#207) and a spec standing for its
  type (#208) are merged.
- **A8c, still to do:** large values spilling to the plane, which
  needs dud#39's binary frames (until then, one call off in-process
  carries 6 MiB encoded), and converging the apps encoder
  (`nt__Encoder`) onto it.

The encoding from the redesign doc's "Task values":
- **Kinds:** data (JSON), tables (Arrow IPC), arrays (`.npy`), bytes
  and file refs. Embedders can register more.
- **Encode in the kernel, decode on the host by the declared type.**
  The encoder generalizes the handler-returns one (`nt__Encoder`): it
  ships to a dud guest as source, and imports numpy, pandas and
  pyarrow lazily.
- **A JSON tree with tagged leaves**, a tag accepted only where the
  declared type allows that kind.
- **Large values spill to blobs in a reserved plane**, with a
  transport for them on each rung.
- **The kernel stub plus host half** that `task` needs: the stub
  encodes before the host call, and passes the value through
  in-process.
- **Classification and the early check:** which kinds a type needs,
  its schema when it is data, and a refusal that names the type, the
  rung and the ways out.

#110 (the return path of app handlers) becomes a special case.

**Exit:**
- Each kind round-trips on in-process, process isolation and dud, and
  process and dud carry exactly the same set.
- A payload whose tags the declared type doesn't allow is refused.
- A large value spills and comes back on every rung.
- The apps suites pass unchanged.

## Phase B: the agex rebuild

### B0. Skeleton

- **Package layout and dependencies:** nontainer at the A3 release
  (raised to A7 later), and `pydantic-ai-slim` with extras pinned to a
  minor range.
- **agex's provider protocol**:
  `stream(turns, tools, settings) → events, final turn, usage`. Its
  first implementation is over `pydantic_ai.direct`, plus a scripted
  provider over `FunctionModel`.
- **v0 of agex's stored run record.** It maps to and from
  pydantic-ai messages at the edge, and every message gets a stable
  id. The streamed events are nontainer's `TurnEvent`, not agex's
  own.
- **CI:** lint, typecheck, and the harness corpus.

**Exit.** CI is green. A scripted provider round-trips a tool call
with its provider id, and the run record round-trips through JSON.

### B1. The session front door, on the scripted model

- `Agent(model, primer, profile)`;
- `agent.session(ws)` with `say` / `asay` / `stream`;
- tools from `nontainer.adapters.tools` (`Toolset`), with
  `PYTHON_UI_NOTE` appended when the host renders artifacts;
- each run inside `ws.turn`;
- inbox delivery through `turn.deliver`;
- runs stored in `__conversation__` with `harness="agex"`;
- `TurnEvent`s yielded directly.

**Exit.** agex yields `TurnEvent`s and passes the harness corpus for
tiers 0-3: the same scenarios, and the same event kinds, that agno
passes.

### B2. Real providers

- Anthropic, OpenAI and OpenRouter first, through pydantic-ai. They
  cover the studio. Google follows;
- streaming;
- thinking and reasoning round trips;
- usage, including cache tokens;
- prompt-cache settings;
- provider-error classification into interrupted vs failed;
- resume in place.

**Exit:**
- **Opt-in live smoke tests**, run only when keys are present:
  - a tool round on each provider;
  - the thinking signature round-tripped on Anthropic;
  - a reasoning round trip on one OpenAI reasoning model.
- The corpus's resume scenarios pass.
- An air-gapped check against an OpenAI-compatible local endpoint.

### B3. Tasks and the outcome host object

The shape is the redesign doc's "Task calls" and "Task values". Four
PRs, with A8 between the first two.

**B3a. Tasks on in-process worlds.** Status: merged (agex #78).
Checked live on the four providers' small models.
- `@agent.task`: typed inputs and a validated return, sync and
  `async def`.
- **`TaskSpec`:** the Python types, the kinds each one needs, and a
  schema for data types.
- **The `task` host object** with `task.success` and `task.fail`, and
  its swappable slot. Agent code sees each argument by name.
- **Inputs are values:** tables as shallow copies, arrays as read-only
  views, other data as deep copies. A live input is bound as a host
  object of the fork, under the world's host-object policy. Narrowing
  one with a `HostObjectGrant` around it waits for `HostObjectGrant`
  itself (the capabilities track).
- **Validation:** strict, by the declared type, with a `TypeError` at
  the call site. A model that stops without finishing is nudged up to
  twice, then the task fails.
- **`Outcome` with `value`**, and the world arguments:
  - a scratch world from `agent.profile` via the memory store;
  - `world=` always forks, through `Store.fork(..., profile=...)`;
  - `keep=`.
- **The check, at call time:** against the world the task runs in. A
  world off in-process is refused before any model call until B3b.

**Exit.** Tests pass:
- a task leaves the caller's world untouched;
- a task that changes its inputs in place leaves the caller's objects
  untouched;
- a live input reaches agent code as it is;
- a validation error is fixed within one script;
- a model that stops is nudged, then the task fails;
- each status: success, failed, cancelled, interrupted;
- a live value comes back in-process;
- a world off in-process is refused before any model call.

**B3b-1. Tasks on every rung.** Status: merged (agex #79, on
nontainer #208). Checked live on the four
providers' small models, in-process and under process isolation.
- **`task` is a stubbed host object:** `HostObject(TaskHost, stub=TaskStub)`.
  The stub lives in `agex.stubs`, standard library only, so a dud
  guest rebuilds it from source. It sends the value to the host half,
  which decodes it by the return type, then stops the script. A value
  that doesn't fit is a `TypeError` at the call, the same on every
  rung.
- **Inputs are typed host data:** `HostObject(value, type=spec)`, by
  value on every rung, a fresh copy each run. A live input is a host
  object as it is, reached through a proxy where the world's code runs
  elsewhere. A value mixing data with live objects, which only
  in-process can carry, gets one copy of its data for the task, its
  live objects shared.
- **Types are nontainer's** (`nontainer.values`): the kinds, the strict
  check of inputs, and the decoding of the value. Each type is
  compiled once with the names it needs and handed to nontainer as a
  spec. pydantic stays only for data types' JSON schemas.
- **The task's types go in `PythonConfig.classes`**, bound by name on
  every rung.
- **What can't cross is refused before any model call, off
  in-process:**
  - a return type with a live part;
  - an input mixing data with a live part;
  - a live object where an input's type allows anything.
- **agex's package exports load lazily**, so a worker importing
  `agex.stubs` doesn't load the loop. That brings a task under process
  isolation down from about 0.5 s to 0.3 s, and on dud from 0.8 s to
  0.4 s.

**Exit.** Tests pass:
- the same value comes back on in-process, process isolation and dud,
  for data, tables, arrays and bytes;
- a refusal reads the same on every rung, at the agent's line;
- a live input is a capability on every rung;
- a live return type, and an input mixing data with a live part, are
  refused off in-process before any model call.

**B3b-2. Needs input, the `__task__` plane, and resuming.** Status:
merged (agex #80). Checked live on the four
providers' small models: told to ask, each asks, then finishes from the
answer.
- `task.needs_input(question)`: the outcome ends `needs_input`, with
  the question as its message. A plain call raises `NeedsInput`, whose
  `outcome.resume(answer)` carries on.
- **Resuming:**
  - a `needs_input` outcome always keeps its world, and `out.ref` names
    it;
  - `grade.resume(ref, answer, world=ws, **live_inputs)` continues it
    by ref, through the world the task ran on, after a restart too on
    a store that persists;
  - a scratch world is kept in the process, and resumes only there;
  - `out.resume(answer)` is the in-process shortcut, reusing the call's
    world and live inputs;
  - the answer is the next message the model reads, after the
    conversation the world holds;
  - a task can ask again, and a resume run to its end drops the world
    unless `keep=True`; one cut short (cancelled, interrupted) keeps it
    waiting, to be resumed again. A first run whose caller is cancelled
    keeps nothing, even after the task asked: no ref reached it.
- **The `__task__` plane:** `spec` (the name, the instructions, each
  type with its schema, and which inputs are stored), `inputs/<name>`
  encoded, `state`, and `value` encoded when it can be. They are
  written in the commits of the task's own runs.
- **Resume with stored inputs.** Encodable inputs come back from the
  plane, decoded by the task's types. A live input isn't stored, so it
  is passed again, and a missing one is refused by name before any
  model call.
- **What a resume refuses:**
  - a world that isn't this task's: by name, inputs and return, the
    signature recorded in its plane;
  - a world reached through another world than the one the task ran
    on (the plane records its session);
  - a world that isn't waiting for an answer;
  - a stored input passed again;
  - an input the task doesn't take.

**Exit.** Tests pass:
- needs input, then resume: in this process, by ref through a kept
  fork, and by ref after a restart on a disk store with a live input
  passed again;
- a resume missing a live input is refused by name;
- asking and resuming work on in-process, process isolation and dud;
- the plane holds the spec, inputs, state and value.

**The brief's live inputs.** Status: merged (agex #81). The brief
describes each live input as agent code uses it: a function by its
signature and docstring summary, any other object by its class's
docstring summary and public methods, up to 20, each with its
signature and summary. Checked live on the four providers' small
models: given a class whose methods they can't guess, each finishes in
one call, where without the description they took four to thirteen
calls exploring the object.

**B3b-3. The shape corpus.** Status: merged (agex #82). 18 scenarios
pass on in-process, process isolation and dud.
- **Task scenarios as data,** as agex's extension of nontainer's
  corpus format, in `agex.conformance`:
  - `shape`: the format. A task's signature with its types as JSON
    schemas (`x-kind` for tables, arrays, bytes and live classes), the
    acts (call, resume by outcome or by ref, restart), the model's
    script in neutral task steps, where the scenario runs, and the
    outcome expected;
  - `scenarios`: the corpus; `runner`: `run` and `check`, owning the
    script; `tasks`: `AgexTasks`, agex as a harness;
  - generated as JSON for harnesses in other languages
    (`python -m agex.conformance.export`), with its JSON Schema.

**Exit:**
- Shape scenarios pass:
  - needs input, then resume, including by ref after a restart;
  - a resume missing a live input is refused by name;
  - the same value comes back on each rung, for data, tables, arrays
    and bytes;
  - a live type is refused off in-process.
- The shape corpus JSON is generated, with its drift test.

### B4. Compaction

**Status: done.** agex #84, merged 2026-10-08.
`Agent(compaction=Policy(...))`; `agex.compaction` folds
within a run too, and each reply records the fold its request had
(`Message.fold`), so a request is measured exactly from the latest
report. The six tier 4 scenarios pass; a task folds within its run and
finishes on every rung; checked live on the four providers' small
models with a budget a little above a task's opening request.

agex folds its own conversations, writing nontainer's records (A4):
its own decision, splice and summary, using the core's helpers where
they fit. It owns its loop, so it can send the model a folded history
while storing every message.

**Folding within a run.** Folds in agno's adapter cut only between
runs, which leaves one long run unbounded. For agex that is the main
case: a task is one run, and a long one is what outgrows the window.
agex folds the run in progress too:
- at a step boundary, never between a tool call and its result;
- keeping the latest steps as they are;
- with the fold's anchor in the run, so the record and its rules are
  unchanged.

**Chaptering waits for curation's trace projection.** That projection
is what makes `/chapters` browsable. A chapter is a named fold with
`first` set, using the agex chapter prompt, so it adds no new
mechanism. It joins after the projection lands, which may be after the
shape freeze.

**Exit.**
- The harness corpus's compaction scenarios pass on agex.
- A task whose run outgrows its budget folds within the run and
  finishes, in-process, under process isolation and on dud.

### B5. Delegation

Re-scoped 2026-10-08: side effects through the `sessions` tool (B5a),
functions as agent-defined tasks (B5b). Code-level `ask(returns=,
agent=)` is dropped; the redesign doc's "Agent-defined tasks" has the
why.

#### B5a. The `sessions` tool

**Status: merged (agex #86).**
- `agent.session(ws, sessions=True)`: the session delegates through
  its `sessions` tool, and each delegate is an agex session of the
  same agent on a fork (`agex.delegation.Runner`, an async
  `SessionRunner`), with the parent's profile.
- A delegate answers once nothing it waits on is outstanding
  (`auntil_settled`), within `max_wakes`; `wake()` runs a turn that
  opens with answers that landed between turns.
- The helper is built by the first turn, on that turn's loop.

**Exit (met).** Tier 5 scenarios pass on agex, and a live delegate
round trip passes.

#### B5b. Agent-defined tasks

**Prerequisites,** each its own small PR, in dependency order:
- **Clean context for host-side async work** (nontainer). In-process,
  a coroutine scheduled from a host call inherits the sandbox's
  context variables, so its network is denied. Run host-side work in a
  fresh `contextvars.Context()`.
- **Waiting host calls don't count against the timeout.** sandtrap
  moves its checkpoint's start time forward by the call's duration,
  in-process and in the process worker; dud's supervisor pushes its
  deadline back by the relay time; nontainer marks a `HostObject` as
  one that waits. Releases go sandtrap, then dud, then nontainer.
- **dud #40:** a dataclass defined in guest code fails, since
  `__dud__` isn't in `sys.modules`.
- **Spec export** (nontainer): a `values.Spec` written out as data and
  read back, records and enums included, without evaluating anything.

**agex:**
- **The `agex` host object:** a stub in the kernel, a host half holding
  the embedder's `Agent`, bound to the world it serves.
  - `@agex.task` and `@agex.task(primer=)`; the call; `.map`, bounded.
  - Refusals at the decorator: a lambda, a missing docstring or
    annotation, code in the body, a live type off in-process.
  - `TaskFailed`, and `TaskNeedsInput` carrying the question.
- **Shapes:** the stub sends the spec as data; the helper's world gets
  generated classes; the stub decodes the result by the caller's own
  annotations.
- **The helper's world:** scratch, from the caller's profile without
  `agex`.

**Exit:**
- Shape scenarios pass on every rung:
  - an agent-defined task returns the caller's own record type, with a
    class defined in the script;
  - nested records, enums, tables and arrays come back as declared;
  - each refusal at the decorator;
  - `.map` returns in order, and runs at once;
  - helper time doesn't end the caller's script under a short timeout.
- A live round trip under agex's loop, and one under agno with
  nontainer, from the same host object.

**Likely nontainer follow-ups** (0.10.x, as B5 finds them):
- `SessionRunner` typed for an `async def run` (agex casts today);
- the open question carried from A5: a fork taken mid-`run_python`
  splits that call's commit. B5b avoids it with scratch worlds.

### B6. The studio seam, designed against both loops

- **A `TurnDriver` protocol plus `DriverSpec`**, yielding `TurnEvent`
  (`harness.md`, "The studio's `TurnDriver`"). `turns.py` is
  rewritten over it.
- **The agno driver** wraps `arun`, `acontinue_run`, `acancel_run`
  and `ais_cancelled`, maps agno's `RunEvent`s onto `TurnEvent`, and
  moves `on_delivered` and `on_fold` into the stream. The `FakeAgent`
  tests move under it.
- **The agex driver** passes agex's `TurnEvent`s through. It sits
  behind the studio's loop knob (for example
  `NONTAINER_STUDIO_LOOP=agex`), and agno stays the default.

**Exit:**
- The studio's pinned behaviors (test_server, test_delegates,
  test_e2e) are parametrized over drivers and pass on **both**.
- Then dogfood: real sessions on agex, including delegation and a
  long session that folds.

### B7. Debug and bench

`agex.debug` (view, token counts, pretty printing) and `agex.bench`,
ported off the core surface. This can slip past the freeze.

## Phase C: the shape freeze

- Write the shape doc, with each Python signature next to its TS
  counterpart.
- Pin the harness-corpus and shape-corpus versions.
- Async agent code tier A must be done, so "scripts may await" holds
  on both sides. That is a parallel track, below.
- Release agex.

**Exit.** A tagged freeze: doc and corpus versions that TS can vendor.

## Phase D: the TS catch-up

Outline only; it gets its own plan at the freeze:
- agex-ts rebuilt to the frozen shape;
- a TS world package with nontainer's semantics, limited to what a
  serverless studio needs;
- agex-studio gets a third `KernelAdapter`, with old sessions staying
  on the `ts` kernel.

## Parallel tracks

None of these blocks Phase B except where noted.

- **Async agent code** (`async.md`):
  - sandtrap #55, the move to the `PyCF_ALLOW_TOP_LEVEL_AWAIT` flag,
    and a sync entry point that accepts `await`;
  - nontainer tier A, including the `asyncio` grant, tool text and
    hints;
  - the dud runner.

  It must land before Phase C.

  **Status (2026-10-08): not started.** sandtrap #55 is open; only its
  first half shipped (top-level `await` comes back as an error result
  instead of raising). Proposed: after 0.10.0, as its own round,
  released in dependency order (sandtrap, then dud, then nontainer).
  B5 and B6 don't need it: fan-out from code is a task's `.map`, run
  on the host.
- **Capabilities** (the redesign's "Capabilities belong to the
  world"):
  - `HostObjectGrant`;
  - the host-surface index and `help()` metadata;
  - `provide` semantics: recursive by default, `unsafe=`, embedded
    skills auto-mounted;
  - scopes checked at call time (nontainer #143).

  agex v1 works without them. The host index makes host objects usable
  without hand-written primers.
- **Curation** (`docs/curation.md` steps 2 onward). It starts after
  A2: the trace projection reads `__conversation__/index`, and
  `TraceSource` picks a renderer by harness. agex's chaptering follows
  the projection.
- **nontainer #110**, the typed codec for the return path of app
  handlers. A8 generalizes it, and B3b and B5 need A8.

## Risks

- **A2's migration touches real stores, production ones included.** A
  dry run on copies is a hard exit condition. Downgrading is not
  possible after a migration write, so the release notes have to say
  so.
- **"No behavior change" across A1-A6 is a lot of surface.** The
  studio's pinned tests and the corpus are the safety net. Where a
  step can't hold the line, it names the change as a fix.
- **Phase B's speed tempts shape drift before the freeze.** Old agex
  went through four breaking rebuilds in about four months. Each
  shape decision lands in the redesign doc's log as it is made.
- **pydantic-ai releases fast.** Keep it to a pinned minor range, and
  keep the provider seam small.

## Open questions

None at the moment. New ones go here. Decisions go to the redesign
doc's log.
