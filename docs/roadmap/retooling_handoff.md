# Retooling — Session Handoff

Handoff note for resuming the retooling work in a fresh session.
Companion to `retooling.md` (design) and `retooling_plan.md` (phases).

## Where we are

- **Branch:** `nxt-retool` (based on `nxt`)
- **Phase 1: done** — commit `f122813`
- **Phase 2: pending** — this is where to start

## Recommended first moves

```bash
git checkout nxt-retool
git log --oneline nxt..HEAD
```

Then read, in order:

1. `docs/roadmap/retooling.md` — the design (why/what)
2. `docs/roadmap/retooling_plan.md` — the phased implementation (how)
3. This file — delta from where Phase 1 landed

## What Phase 1 put in place

New module `agex/agent/emissions.py` with six dataclasses:

- `TextEmission`, `ThinkingEmission` — content blocks (thinking has optional `signature`, `redacted` flag for Claude's `redacted_thinking`)
- `PythonEmission`, `TerminalEmission` — actionable tool calls with optional `title`, `signature`
- `FileWriteEmission`, `FileEditEmission` — file operations, superseding `FileAction` / `EditAction`
- Union alias `Emission` and tuple `ACTION_EMISSION_TYPES` for isinstance checks

`ActionEvent` reshaped in `agex/agent/events.py`:

```python
class ActionEvent(BaseEvent):
    emissions: list[Emission] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
```

Old named fields (`title`, `thinking`, `report`, `code`, `terminal`, `file_actions`) are gone. Repr methods rewritten to iterate emissions — they work but are placeholder-quality.

`PrintAction` promoted from tuple subclass to a `@dataclass(args: tuple, emission_id: str | None = None)` that still supports `iter/len/getitem` so legacy joining code keeps working.

`ImageAction` got `emission_id: str | None`, pickled through `__getstate__` / `__setstate__`.

`TokenChunk` in `agex/llm/core.py`:

- Added `emission_index: int = 0`
- Replaced `"report"` with new `"text"` type (user-facing prose)
- Removed `"edit"` and `"file_action"` types and the `action` field (old XML shape)

## What's still wired to the old shape (Phase 2 targets)

These files reference the old ActionEvent fields or old TokenChunk shape and need reshaping:

- `agex/llm/core.py` — `LLMResponse` still has `title`/`thinking`/`code`/`file_actions`; `ResponseBuilder` is ~300 lines of XML-era parsing. Replace with an emission-list response + builder.
- `agex/llm/formats/tool_use/parser.py` — currently emits `file_action` tokens with bundled FileAction objects. Rewrite to emit per-emission tokens with `emission_index`.
- `agex/llm/formats/tool_use/renderer.py` (~426 lines) — `render_events_as_tool_use` walks `event.file_actions` + main action. Rewrite to walk `event.emissions` in order.
- `agex/agent/loop/event_factories.py::create_action_event` — builds old-shape ActionEvent from LLMResponse. Shorten to `ActionEvent(agent_name=..., emissions=response.emissions, ...)`.
- `agex/agent/loop/sync_loop.py`, `async_loop.py`, `mixin.py` — main execution loop. Rewrite to iterate `event.emissions` sequentially with shared Python namespace and terminator early-exit.
- `agex/eval/bridge/result.py` — propagate `emission_id` to PrintAction/ImageAction parts; drop the `TaskContinue` branch.
- `agex/agent/loop/common.py` — re-export list still mentions `TaskContinue`.

## Deferred stubs still in the tree (delete in Phase 2 when safe)

- `FileAction`, `EditAction` in `agex/agent/datatypes.py` — marked deprecated. Still imported by core.py, parser.py, renderer.py, and several test files. Drop after those are rewritten.
- `TaskContinue` in `agex/agent/datatypes.py` — same story. The `task_continue` builtin also needs removal from wherever it's registered in the sandbox (check `agex/eval/builtins/` or similar).

## Known design decisions (don't rethink these)

From the resolved open-questions section of `retooling.md`:

1. **Streaming:** `emission_index` on TokenChunk, not `emission_start` / `emission_end` markers.
2. **XML:** delete entirely (Phase 3). Keep `WireFormat` protocol interface as a seam.
3. **OutputEvent:** stays (not an emission). `emission_id` lives on each *part* (PrintAction, ImageAction), not on the event.
4. **Reasoning effort:** no agex-level abstraction. Use `**kwargs` pass-through. Consider setting `OpenAI` client default to `reasoning={"effort": "low"}` for agentic use.
5. **Report → TextEmission:** user-facing prose is now native assistant text, no schema param.
6. **Title:** per-emission `title` field (optional). UI shows most recent non-empty per turn.
7. **Interleaved thinking:** ThinkingEmission sits in order among other emissions; no special-casing.
8. **Multi-action per turn:** permitted in the schema. Sequential execution, shared Python namespace. Early-exit on first terminator (`task_success` / `task_fail` / `task_clarify`). Remaining emissions logged but not executed.
9. **`task_continue` is gone.** Python completing normally = implicit continue.

## Task tracker IDs

Retooling tasks (all on current task list):

- **#61** Phase 1 — **completed**
- **#62** Phase 2: Renderer + execution loop — **pending** (this is next)
- **#63** Phase 3: Wire format + delete XML — pending
- **#64** Phase 4: Provider adapters + signature round-trip — pending
- **#65** Phase 5: Cleanup sweep + docs — pending
- **#66** Phase 6: agex-studio follow-up — pending

Mark #62 `in_progress` when starting.

## Suggested Phase 2 order (within the phase)

1. **Reshape `LLMResponse`** to hold `emissions: list[Emission]` + usage fields only. Delete everything else.
2. **Replace `ResponseBuilder`** with an `EmissionsBuilder` that assembles emissions from TokenChunks. Group tokens by `emission_index`. Key decisions:
   - An emission begins when you see a TokenChunk with a new `emission_index` OR a new `type` within the same index (title → thinking → code all belong to one PythonEmission).
   - Build the right emission type based on which tokens arrived: `title`/`thinking`/`code` → PythonEmission; `title`/`commands` → TerminalEmission; `text` alone → TextEmission; `thinking` alone → ThinkingEmission; `file` content → FileWriteEmission/FileEditEmission.
3. **Rewrite tool-use parser** to emit tokens with correct `emission_index`. Each tool_call_start bumps the index. Text blocks (from Anthropic/Gemini content arrays) also get their own index.
4. **Rewrite tool-use renderer** to walk `event.emissions` → build ordered content blocks. This is where signature round-trip will hook in later (Phase 4).
5. **Rewrite `create_action_event`** — trivial once LLMResponse holds emissions.
6. **Rewrite execution loop** (sync + async) — sequential emission walk with shared namespace and terminator early-exit. Propagate emission_id to OutputEvent parts.
7. **Drop `task_continue`** — remove the builtin, the `TaskContinue` exception, and its handling in `bridge/result.py`.
8. **Test sweep** — many tests will break. Rewrite the ones worth keeping (most of `test_tool_use_renderer.py` can adapt; XML-specific tests stay broken until Phase 3 deletes them).

Phase 2 checkpoint: `examples/funcy.py` should run end-to-end against a real provider (e.g. Sonnet via OpenRouter) and complete at least one prompt successfully.

## Files worth reading before starting

- `agex/agent/emissions.py` — Phase 1 output, the new core type
- `agex/llm/formats/tool_use/parser.py` — clean starting point for the rewrite
- `agex/llm/core.py::ResponseBuilder` — understand what it currently does before replacing
- `docs/roadmap/retooling_plan.md` — the full Phase 2 spec

## Nothing else is blocking

Phase 1 is a fully-committed, self-contained piece. Nothing is half-done on disk. The branch is `nxt-retool`; rebasing onto the latest `nxt` should be clean if `nxt` moves.
