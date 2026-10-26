# agex plan

What is done and what comes next. The shape and the reasons for it are
in `docs/design.md`.

**Status (2026-10-12):** agex 0.13.0 is released, the first release of
the rebuild. Next are the docs and studio follow-ups, then the async
agent code round, then the shape freeze for TS.

## Ground rules

- **One step is one PR,** or a short stack. The user drives merges and
  releases.
- **Dependencies release in order,** lowest first (sandtrap, dud,
  nontainer, then agex), each user raising its floor after.
- **nontainer stays safe at every step.** Its suite and the studio's
  pass unchanged unless a step names a fix, since deployments run on
  nontainer.
- **Each step's exit names the corpus scenarios that pin it,** where
  the step changes behaviour the corpora cover.
- **Python >= 3.10, `uv run pytest`, Conventional Commits.**

## Done

The rebuild ran in two phases: nontainer and the studio grew the seams
a second loop needs (A), and agex was rebuilt as a loop over them (B).

| step | what | PRs |
|---|---|---|
| A0 | small prerequisites: an ephemeral store, `nontainer.Profile` | nontainer #196, studio #88 |
| A1 | the neutral tool layer: each tool defined once | nontainer #198 |
| A2 | the harness-neutral conversation index and plane, with migration | nontainer #199 |
| A3 | the turn API (`ws.turn`), `TurnEvent`, the corpus machinery | nontainer #200, #201, #204 |
| A4 | the compaction contract: nontainer keeps the record of a fold | nontainer #209 |
| A5 | delegation: async runners, landing without the parent's lock | nontainer #210-#214, studio #90 |
| A6 | early studio fixes, on agno | studio #91 |
| A7 | the conformance kit ships (tiers 0-5) | nontainer 0.10.0 |
| A8a-b | the value encoding (`nontainer.values`) and its transport on each rung | nontainer #205-#208 |
| B0-B1 | a clean slate, the provider protocol and run record, the session loop | agex #71-#74 |
| B2 | real providers: OpenRouter, Anthropic, OpenAI, Google, local endpoints | agex #75, #76 |
| B3 | tasks: worlds, every rung, asking and resuming, briefs, the shape corpus | agex #78-#82 |
| B4 | compaction: agex folds its own conversations, within a run too | agex #84 |
| B5a | delegation through the `sessions` tool | agex #86 |
| B5b | agent-defined tasks, in apps too, with a stand-in | agex #88-#90; nontainer #222-#230; sandtrap #59, #60; dud #43 |
| B6 | the studio on both loops: `TurnDriver`, tool parity, three dogfood rounds | studio #92-#97, #104, #105, #107; agex #92-#102; nontainer #231-#235; dud #44; sandtrap #61 |
| — | agent-defined tasks in the studio | agex #103, studio #108 |
| — | docs and the 0.13.0 release | agex #107-#109 |
| — | agents working with your code (docs); a task call typed by its return type (unreleased) | agex #111 |
| — | `AgexTraces`: an agex session read as a trace (unreleased) | agex #? |

Releases along the way: nontainer 0.9.0 to 0.11.0, sandtrap 0.4.2 to
0.4.4, dud 0.4.2 to 0.4.4, and agex 0.13.0 (2026-10-12). The 0.12 API
is at the `v0.12.4` tag.

## Next

- **Docs:** the rest of `docs/`, one page per topic: sessions, tasks,
  agent-defined tasks, delegation, providers, testing.
- **The splash page:** ashenfad.github.io/agex becomes one static
  `site/index.html` for the ecosystem, deployed by a small Pages
  workflow. It holds agex in a sentence with install and links, the
  ecosystem a line each by layer (agents: agex, nontainer-studio;
  worlds: nontainer, kvgit, monkeyfs; running code: sandtrap, dud,
  termish, reprobate; in the browser: agex-ts, agex-studio), and where
  0.12 lives.
- **The studio on the release:** its agex git source becomes
  `agex>=0.13.0`, and it imports `from agex import cancel_tasks`.
- **The repo description** still has 0.12's tagline.
- **agex #105:** OpenAI reasoning models' summaries, so their thinking
  shows on agex as on agno.
- **B7, debug and bench:** `agex.debug` (a run's view, token counts,
  pretty printing) and `agex.bench`, off the core surface.

## The async agent code round

Agent code may `await` at the top level of `run_python`, the way
IPython allows. Host calls stay sync and run one at a time. It is its
own round, released sandtrap, then dud, then nontainer, and the shape
freeze waits for it. Nothing built so far needs it: fan-out from code
is a task's `.map`, run on the host.

- **sandtrap first** (#55, whose first half shipped: top-level `await`
  under `exec` comes back as an error result). Four items, one PR or a
  short stack, released as **0.5.0**, since the first changes `aexec`
  for its callers:
  1. module semantics: compile with `PyCF_ALLOW_TOP_LEVEL_AWAIT` and
     run `await eval(code, ns)`, in place of the function wrapper;
  2. the worker's async flavor, so process isolation runs it;
  3. a sync entry point that accepts `await`, so host calls are still
     served on the calling thread;
  4. tasks still pending at the end settled inside the sandbox, and
     reported.
- **A stop reaches an awaiting script.** A script parked in
  `await asyncio.sleep(60)` runs no code, so no checkpoint sees a
  cancel. The script's private loop gets a cancel hook, with a
  stop-mid-`await` test on every rung.
- **The dud runner** compiles with the same flag and runs a coroutine
  through `asyncio.run`.
- **nontainer:** a narrow `asyncio` grant (no loop handle, `to_thread`
  or subprocess; `TaskGroup` and `timeout` only where the running
  Python has them), tool text, and a hint for "`asyncio.run()` cannot
  be called from a running event loop".
- **Out of the round:** async host objects (they wait for an embedder
  that needs them) and async app handlers (nothing needs them).

## Phase C: the shape freeze

- Write the shape doc, each Python signature beside its TS counterpart.
- Pin the harness-corpus and shape-corpus versions.
- Record that host calls stay sync in Python agent code and are awaited
  in TS, where every host call is a promise.
- Release the frozen shape: 0.14.0 if it changed anything since 0.13,
  otherwise a 0.13.x.
- Decide the studio's default loop (below).

**Exit:** a tagged freeze, with doc and corpus versions TS can vendor.

## Phase D: the TS catch-up

Outline only; it gets its own plan at the freeze:
- agex-ts rebuilt to the frozen shape;
- a TS world package with nontainer's semantics, limited to what a
  serverless studio needs;
- agex-studio gets a third `KernelAdapter`, with old sessions staying
  on the `ts` kernel.

## Parallel tracks

None of these blocks the freeze.

- **Capabilities** (design, "Capabilities belong to the world"):
  - the host-surface index and `help()` metadata, so a host object
    needs no hand-written primer, and `help(db)` shows the class under
    process isolation and on dud as it does in-process;
  - `HostObjectGrant`, honoured by every transport;
  - scopes checked at call time (nontainer #143);
  - `provide`: recursive by default, `unsafe=`, embedded skills
    mounted.
- **A8c:** large values spill to the plane, and the apps encoder moves
  onto the shared one. It needs dud #39's binary frames; until then one
  call off in-process carries about 6 MiB encoded. It changes how a
  value travels, not its shape, so it can follow the freeze.
- **A3c:** harness host objects, attached after a world opens. Deferred
  (design, decisions, 2026-10-07): a task's fork gets `task` when it
  opens, and nothing needs more yet.
- **Curation** (nontainer `docs/curation.md`, steps 2 onward): the
  trace projection reads agex's runs through `agex.traces.AgexTraces`
  (nontainer 0.11.6), and agex's chaptering follows it. Whether agex
  stores its system prompt, which a trace would show, is open.
- **Parked:** the harness running in the browser with a server as a
  kvgit remote; a remote `KernelAdapter` letting agex-studio drive a
  nontainer-studio server; world-format interop between Python and TS.
  None is needed until a session has to cross languages.

## Risks

- **Shape drift before the freeze.** Old agex went through four
  breaking rebuilds in about four months. Each shape decision lands in
  the design's decisions as it is made, and the freeze pins the rest.
- **Two loops double the studio's edge cases** (cancel, resume,
  retries, compaction). Its tests run on both loops to keep that honest.
- **pydantic-ai releases fast.** Keep it to a pinned minor range, and
  the provider seam small.

## Open questions

- **The studio's default loop.** agno is the default, and agex sits
  behind `NONTAINER_STUDIO_LOOP=agex`. Whether agex becomes the
  default, and when agno goes, is decided at the freeze.
- **MCP servers.** The tool protocol is MCP's, so an MCP client could
  hand agex a server's tools; whether and when is open.

More, on the shape itself, are in the design's open questions.
