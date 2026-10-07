# agex rebuild: implementation plan

Status: in progress. Phase A is done through the turn API (nontainer 0.9.1, 2026-10-06), and Phase B is starting.

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
  minor, which 0.x allows, and the user is its only real user. Use
  pre-releases (`0.13.0a1`, …) during Phase B and `0.13.0` at the
  shape freeze. The CHANGELOG says plainly that it is a new API. Work
  on a fresh branch from `main`, not `nxt`.

## The stages

```
A0 small bits ─┐
A1 tool layer ─┼─► A3 turn API + TurnEvent + corpus ─► B0 skeleton ─► B1 sessions ─► B2 providers ─► B3 tasks
A2 conv. index ┘          │                                                                         │
                          ├─► A4 compaction core ─────────────────────────────► B4 compaction ◄─────┤
                          ├─► A5 delegation ──────────────────────────────────► B5 delegation ◄─────┤
                          └─► A7 kit ships (with A4/A5's scenarios)                                 │
A6 early studio fixes (agno only; any time)                                                         │
                                       B6 studio seam, both loops (needs B1-B3) ◄───────────────────┘
                                                              ─► C freeze ─► D TS
```

Only A0-A3 come before agex starts. A4, A5 and A7 run alongside
B1-B3. The studio's `TurnDriver` seam is built in B6, against both
loops, rather than in Phase A.

## Phase A: nontainer and the studio

Each step follows `harness.md`'s build order.

### A0. Small prerequisites (before B0)

**Status: nontainer #196 merged 2026-10-07; studio #88 still open.**
- nontainer #196 (branch `feat/a0-ephemeral-store-env`): the memory
  store, `Profile` (first built as `Env`) and the agno instruction
  fix. Unreleased on main.
- nontainer-studio #88 (branch `fix/resume-comment`): the stale
  comment.
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

**Status (2026-10-06): nontainer #199 open.**
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
- **The studio's four call sites** go in a follow-up studio PR after
  the nontainer release. Until then, the studio stays on the released
  nontainer (it imports `CONVERSATION_SESSION_KEY`).

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

**Status (2026-10-06): split in two.**
- **A3a: nontainer #200 open.**
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
- **A3b: nontainer #201 open.**
  - **What it adds:**
    - `ws.turn` / `ws.turns.begin` / `turn.end` (`nontainer.turns.Turn`), with one commit per turn stamped `{"tool": "turn", "runs": {id: status}}`.
    - `TurnInProgress`.
    - The agno db and `end_turn` stage under an open turn.
    - `adapters.agno.finish_turn` / `status_of_error`.
  - **Gaps:** `AgnoHarness` has no known gaps left.
  - **The studio:** its suite passes unchanged (it opens no turns yet).
- **Split out as A3c:** harness host objects (host objects are fixed at open via PythonConfig today) and the profile fingerprint in the turn stamp (manifest contents undecided). Neither blocks a loop that owns its turns; agex needs host objects by B3.
- **Milestone release:** due once A3b merges, as A1-A3b. A3c can follow in a later release.

### A4. The compaction algorithm moves into core

`turn.context(messages, input_tokens=, summarize=, policy=)` runs over
a neutral `Msg`. The agno compaction adapter shrinks to the message
mapping, the token-report quirks and `summarize`. Fix
`docs/compaction.md`.

**Exit.** Compaction scenarios pass on agno:
- a fold over budget;
- the fold in force spliced in;
- no summary in stored runs;
- a rewind takes back its folds;
- a fresh fork drops them.

The studio's compaction tests pass.

### A5. Delegation

- `sessions.until_settled(run_turn, max_turns=)`;
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
  `asyncio.Condition`. A5's async runners remove those loops later,
  but the sink should be safe regardless.
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

- `@agent.task`: typed inputs, validated returns with a retry on
  mismatch, sync and `async def`.
- **The `task` host object** (`task_success` / `task_fail` / needs
  input), attached as a harness host object, with its swappable slot
  and the durable outcome key in a reserved plane.
- **`Outcome`, and the shape of world arguments:**
  - a scratch world from `agent.profile` via the memory store;
  - `world=` always forks;
  - `keep=`;
  - needs input followed by `resume`.
- **The shape corpus begins**, as agex's extension of the corpus
  format.

**Exit:**
- Shape scenarios pass:
  - a task leaves the caller's world untouched;
  - a validation error is fixed within one script;
  - needs input, then resume;
  - each outcome status;
  - the value comes back on in-process isolation.
- Process and dud value returns wait for nontainer #110's typed
  codec and are marked pending.

### B4. Compaction

agex's compaction adapter runs over A4's core.

**Chaptering waits for curation's trace projection.** That projection
is what makes `/chapters` browsable. A chapter is a named fold with
`first` set, using the agex chapter prompt, so it adds no new
mechanism. It joins after the projection lands, which may be after the
shape freeze.

**Exit.** The compaction scenarios pass on agex.

### B5. Delegation from code

- agex as an async `SessionRunner`;
- a code-level `ask` / `wait` harness host object;
- `ask(returns=)`;
- `ask(agent=other)`, inheriting the caller's environment;
- the delegate's typed value on `Answer`, which also needs #110 off
  in-process.

**Exit.** Tier 5 scenarios pass on agex. The shape corpus covers the
replacement for spawn: fan-out, a fresh view, and typed results.

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
- **nontainer #110**, the typed codec for view-exec returns. B3 and B5
  need it to return values off in-process.

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
