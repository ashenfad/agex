# Retooling: align the agex turn with provider-native tool use

!!! success "Implemented"

    Landed on branch `nxt-retool` ahead of 0.11.  See the
    `[Unreleased]` block in [CHANGELOG.md](https://github.com/ashenfad/agex/blob/main/CHANGELOG.md) for the
    shipping summary; package READMEs
    (`agex/llm/formats/tool_use/README.md` for the wire format,
    `agex/llm/README.md` for per-provider reasoning) carry the
    implementation-level reference.  This page is kept as historical
    design context.

## Summary

Reshape the core event model, execution loop, and primer to match how
frontier LLMs actually emit turns. Eliminate agex-specific ritual
(task_continue, XML narration, single-main-action) in favor of a flat
list of ordered emissions per turn, with native thinking + signature
round-trip. Net effect: agex becomes a thinner translation layer, and
model tool-use fluency should improve because we stop asking the model
to learn our dialect.

## Why now

Three concrete pain points on the current shape:

1. **Gemini 3 rejects our requests.** Every function_call response now
   returns a `thought_signature` that must be round-tripped verbatim on
   subsequent turns. We don't capture or replay it → 400 on any turn
   after the first.
2. **Claude 4.6 adaptive thinking interleaves thinking blocks between
   tool_calls.** Our `ActionEvent(title, thinking, code, file_actions)`
   flat shape can't represent `[thinking, tool_use, thinking, tool_use]`
   without losing information.
3. **Our primer teaches agex dialect.** The `<THINKING>` narration
   requirement, `task_continue()` discipline, and single-Python-block
   constraint are all OOD for the training distribution of modern
   frontier models. Every deviation from the native tool-use idiom
   costs fluency.

These aren't independent bugs — they're symptoms of a shape mismatch
between what we built (pre-Claude-extended-thinking, pre-Gemini-3) and
what providers now emit.

## Proposed changes

### 1. ActionEvent becomes an ordered list of emissions

Replace the named-field shape:

```python
ActionEvent:
  title: str
  thinking: str
  report: str
  code: str | None
  terminal: str | None
  file_actions: list[FileAction | EditAction]
```

With an ordered emission list:

```python
ActionEvent:
  emissions: list[Emission]     # what the model produced, in order

Emission = (
  TextEmission        # user-facing prose (was "report")
  | ThinkingEmission  # agent's own reasoning, signature-bearing
  | FileWriteEmission # path, content, mode, optional title, signature
  | FileEditEmission  # path, search, op, content, optional title, signature
  | PythonEmission    # code, optional title, signature
  | TerminalEmission  # commands, optional title, signature
)
```

Everything the model emits becomes an ordered emission. No more
top-level named fields on ActionEvent — `title`, `thinking`, `report`,
`code`, `terminal`, `file_actions` all dissolve into emission entries.

**Text vs. thinking — distinct concerns:**

- `TextEmission` is user-facing prose. Maps to Anthropic text blocks,
  Gemini `Part(text=...)`, and OpenAI assistant `content`. In native
  tool use the model already emits text blocks intermixed with
  tool_use — no primer dialect needed; it's literally free. Replaces
  the old `report` field. Multiple per turn is idiomatic (progress
  narration between tool calls).
- `ThinkingEmission` is the agent's own reasoning trace for its
  future self. Maps to provider thinking blocks (Claude, Gemini) with
  signature round-trip. For non-native-thinking providers, falls back
  to narration-via-schema.

**Title becomes a per-emission label.** Each tool-call-shaped emission
(Python, Terminal, FileWrite, FileEdit) carries an optional `title`
string — a short UI label for that specific step. A turn with
`[write lib.py, write tests.py, run tests]` naturally has three
titles ("Creating helpers", "Writing tests", "Running tests"). UI
shows the most recent non-empty one live; historical chat shows the
last per turn.

**Why ordered:** Claude adaptive thinking interleaves thinking blocks
between tool_calls, and Gemini 3's `thought_signature` belongs to a
specific function_call in a specific position. Reconstructing an
assistant turn faithfully requires preserving emission order.

**OpenAI Chat Completions constraint:** OpenAI's assistant message is
`content: str | null` + `tool_calls: list`. Text can't interleave
with tool_calls on the wire. For OpenAI we concatenate all
TextEmissions into `content` (losing interleaved position relative to
tool_calls). Acceptable because OpenAI has no native thinking to
interleave *with* — no structural loss in practice. If we ever move
to OpenAI's Responses API, ordered interleaving becomes possible.

### 2. Drop task_continue; terminators become the only explicit loop control

Current: agent must call `task_continue()` (or a terminator) to end
each turn cleanly.

New: Python execution completing normally *is* an implicit continue.
Only the terminators remain:

- `task_success(result)` — task is done, here's the result
- `task_fail(message)` — task is done, it failed, here's why
- `task_clarify(message)` — task is blocked, caller input needed

Rationale:
- Frontier models are trained on tool-use loops where assistant
  responses naturally end and the loop continues. They don't know
  about special continuation markers.
- Eliminates the `task_continue(huge_value)` prompt-blowup footgun
  entirely — there's no user-controlled observation injection anymore.
- Agex namespace already persists across turns, so "pass data forward"
  is a plain Python variable assignment.

### 3. Multiple emissions per turn

Current: one Python OR one Terminal, plus any number of file actions.

New: any number of any emission type, executed sequentially in the
order they appear. Python emissions share namespace (as if
concatenated). Early-terminate on first terminator (subsequent
emissions get logged but not executed).

What this enables:
- Claude interleaved thinking structurally preserved
- Gemini 3 parallel function_calls structurally preserved (we still
  execute sequentially for Python-namespace safety)
- Pre-planned multi-step turns (`write lib.py → write tests.py →
  python(run tests)`) without per-step round trips

What this doesn't enable:
- Mid-turn feedback loops. The model can't branch on a tool result it
  hasn't received yet. `bar.py based on python's output` still needs
  a new turn. Multi-emission is about structural fidelity, not new
  capability.

### 4. Provider-native thinking + signature round-trip

Add a flag on `ToolUseWireFormat`:

```python
ToolUseWireFormat(native_thinking: bool = False)
```

When `native_thinking=True`:
- Drop the `thinking` parameter from the `python_action` /
  `terminal_action` tool schemas
- Drop the THINKING section from the primer
- Expect thinking to arrive as ThinkingEmissions from the provider
  stream
- Each emission carries an optional `signature: bytes | None` the
  client is responsible for capturing + replaying

Client wiring:
- `Gemini` / `PyfetchOpenAI`-over-Gemini: `native_thinking=True`,
  capture `thought_signature` per function_call
- `Anthropic` / `PyfetchAnthropic` (4.6+): `native_thinking=True`,
  capture thinking-block signatures + redacted_thinking blocks
- `OpenAI` GPT-5.x: `native_thinking=True` for suppressing our prompt
  (server-side reasoning is hidden, no round-trip), plus passthrough
  knob for `reasoning_effort`
- `OpenAI` GPT-4o-class, `Anthropic` pre-4.x: `native_thinking=False`,
  keep our narration-in-schema approach

### 5. Signature storage is narrow, not opaque

Each emission gets an optional `signature: bytes | None`. Not a
`provider_state: dict` grab-bag — those tend to grow.

For Gemini, per-function_call thought_signature sits on the emission
it belongs to. For Claude, thinking-block signatures sit on
ThinkingEmissions. The position of emissions in the ordered list
preserves the "first function_call in the step" requirement for
Gemini automatically.

### 6. Primer slim-down

With the ritual gone, the primer gets dramatically leaner:

- No XML-mode primer: `XmlWireFormat` stays available but becomes a
  legacy option for older models. New primer lives on
  `ToolUseWireFormat`.
- No task_continue section
- No "wrap in THINKING" section for native-thinking providers
- No "single python block" discipline
- No "emit a report" section — user-facing text is just native
  assistant text, no schema parameter

Expect ~60% reduction in primer size for native-thinking providers.

## What the execution loop looks like

Pseudocode for one turn:

```
emissions = parse_assistant_response()  # ordered list
for emission in emissions:
    match emission:
        case ThinkingEmission:
            pass  # logged, not executed
        case FileWriteEmission | FileEditEmission:
            apply_to_fs(emission)
            record_tool_result(emission, synthesized_result)
        case PythonEmission:
            result = execute_in_sandbox(emission.code)  # shared namespace
            record_tool_result(emission, rendered_observations)
            if result.terminator:
                break  # subsequent emissions logged but not executed
        case TerminalEmission:
            result = execute_shell(emission.commands)
            record_tool_result(emission, output)
```

## Migration

Clean break. The user is still the sole real user; breaking changes
are acceptable. Not doing:

- No ActionEvent v1 ↔ v2 reader
- No task_continue stub that emits a deprecation warning
- No dual-path renderer

Doing:

- Single atomic reshape across event model, renderer, wire format,
  execution loop, primer
- Bump minor version
- Update CHANGELOG with a "you will need to regenerate any pickled
  event logs" note

## What stays unchanged

- Wire-format protocol (`WireFormat` interface)
- LLM client public APIs (OpenAI, Anthropic, Gemini, Pyfetch variants)
- Task / agent / fs registration APIs
- Sandbox evaluation semantics (still real Python, still namespace-
  preserving across turns)
- Event log storage (just new event shapes inside)
- All non-action events (TaskStartEvent, OutputEvent, SuccessEvent,
  FailEvent, ClarifyEvent, ChapterEvent, FileEvent, SystemNoteEvent)

## Resolved design questions

### Streaming emission boundaries

**Decision:** extend TokenChunk with `emission_index: int` rather than
introducing `emission_start` / `emission_end` marker chunks.

TokenChunk's existing `(type, content, done)` protocol already encodes
field boundaries. A new-emission-of-same-type is detectable via a
`(type, done=true)` followed by a new `(type, done=false)`. The only
missing dimension is *which* emission a chunk belongs to when the turn
has several — `emission_index` fills that in.

- 0-based, increments monotonically per assistant turn
- 0 for non-emission chunks (`output`, `error`, framework signals)
- Resets per ActionEvent
- UI consumers group by `emission_index` to build per-emission panels

New token type needed: `text` for TextEmission. Everything else
(`title`, `thinking`, `code`, `file`, `terminal`) carries over.

### XML wire format

**Decision:** drop the XML implementation entirely. Keep the
`WireFormat` protocol interface as a seam for hypothetical future
non-tool-use encodings.

Reasoning: every frontier provider and all major local inference paths
(vLLM, Ollama, llama.cpp, LM Studio) support OpenAI-compatible tool
use in 2026. Gemma 4 and Qwen3 both train with dedicated tool-use
special tokens. The "fallback for weaker models" use case that
justified XML mode has largely evaporated. Keeping XML working against
the new emission shape is ongoing cost for a path nobody's asking for.

Deleting: `agex/llm/formats/xml/`, XML-mode primer content, XML-mode
tests. Keeping: `agex/llm/formats/wire_format.py` (the protocol), the
`wire_format` kwarg on clients (now always a `ToolUseWireFormat`).

### view_image and execution-side output

**Decision:** keep OutputEvent for execution-side output. Part types
gain `emission_id`.

Emissions are what the model produced at the assistant-turn layer.
OutputEvents, SuccessEvent, FailEvent, ClarifyEvent are what happened
when the runtime executed the emissions. Separating those concerns
keeps the rendering story clean: each tool-call emission gets a
tool_result in the next turn, synthesized from OutputEvents whose
parts trace back to it.

Structural change to support eventual OutputEvent aggregation (for
storage efficiency):

- `ImageAction` gains `emission_id: str | None = None`
- `PrintAction` promoted from tuple-shape to a small dataclass:
  `PrintAction(args: tuple, emission_id: str | None = None)`
- `OutputEvent.parts` unchanged; no event-level emission_id
- Future storage layer can merge consecutive OutputEvents for one
  turn without losing which-emission-produced-what

### Reasoning effort / thinking knobs

**Decision:** no agex-level abstraction. Use existing `**kwargs`
pass-through to the underlying SDK.

Provider semantics aren't actually aligned:

- OpenAI `reasoning.effort`: reasoning-token budget
- Anthropic `thinking.adaptive.effort`: adaptive thinking intensity
- Gemini 3: different config shape, no real opt-out with tools

A unified knob would be lossy for all three. Provider APIs also churn
faster than we should — Anthropic's recent `enabled+budget_tokens` →
`adaptive+effort` migration would've broken an agex abstraction but
passed through untouched via `**kwargs`.

**Idiomatic usage** (documented in client docstrings +
`agex/llm/README.md`):

```python
OpenAI(model="gpt-5.4", reasoning={"effort": "high"})
Anthropic(model="claude-sonnet-4-6",
          thinking={"type": "adaptive", "effort": "high"})
Gemini(model="gemini-3-flash-preview",
       thinking_config=types.ThinkingConfig(...))
```

**Orthogonal to `native_thinking`:** the wire-format flag controls
schema shape ("stop asking the model to narrate"). The provider knob
controls *how hard* the model thinks. Decoupled.

**One policy call worth considering:** OpenAI GPT-5.2+ defaults
`reasoning.effort` to `"none"`. For agex's agentic use case that
default is probably wrong — we likely want to set `reasoning={"effort":
"low"}` (or similar) as a client-level default the caller can
override. Not an abstraction, just a sensible baseline.

## What happens next

If no major objections to the above, the implementation plan splits
roughly:

1. New event dataclasses (ActionEvent + Emission hierarchy)
2. Wire format changes (`native_thinking` flag, primer, schemas)
3. Provider adapters (capture signatures; replay on render)
4. Renderer rewrite (both tool_use and XML paths, scoped to emission
   list)
5. Execution loop (sequential emission execution, shared namespace,
   terminator early-exit)
6. Drop task_continue; update system prompts
7. Tests: rewrite ActionEvent-touching tests; add signature round-trip
   tests per provider; add interleaved-thinking tests for Claude
8. agex-studio follow-up: update history loader, event renderer, and
   primer display
