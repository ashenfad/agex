# Retooling — Implementation Plan

Concrete staged plan for executing the design in
[`retooling.md`](retooling.md). Each phase lands on a working state
with passing tests; between phases we pause, verify, and regroup.
All work happens on a dedicated branch (`nxt-retool` or similar).

The user is the sole real user, so the plan is clean-break: no
compat shims, no old ActionEvent reader, no task_continue deprecation
stub. When the dust settles there's one event shape, one wire format,
one tool-use idiom.

## Phase sequencing at a glance

1. **Data model** — new types, no behavior
2. **Renderer + execution loop** — use the new types end-to-end
3. **Wire format** — `native_thinking` flag, primer slim-down, delete XML
4. **Provider adapters** — signature capture + replay
5. **Cleanup sweep** — delete dead code, update docs, final test run
6. *(separate)* **agex-studio follow-up** — history loader, UI, token router

Expected scope: roughly one working day per phase except Phase 4,
which is the hardest (signature round-trip is provider-specific and
easy to get wrong). Total: ~1 week of focused work before agex-studio.

---

## Phase 1 — New data model

**Goal:** introduce the new types alongside the old ones. Nothing
uses them yet, but they exist and import cleanly.

### New dataclasses

```python
# agex/agent/emissions.py (new module)

@dataclass
class TextEmission:
    """User-facing prose. Maps to Anthropic text blocks / Gemini text
    parts / OpenAI assistant content."""
    text: str
    signature: bytes | None = None   # for providers that sign text

@dataclass
class ThinkingEmission:
    """Agent's internal reasoning. For native-thinking providers,
    carries signature for round-trip. For non-native providers, this
    is the content of the narrated `thinking` schema param."""
    text: str
    signature: bytes | None = None
    redacted: bool = False   # Claude's redacted_thinking blocks

@dataclass
class PythonEmission:
    code: str
    title: str | None = None
    signature: bytes | None = None

@dataclass
class TerminalEmission:
    commands: str
    title: str | None = None
    signature: bytes | None = None

@dataclass
class FileWriteEmission:
    path: str
    content: str
    mode: Literal["write", "append"] = "write"
    title: str | None = None
    signature: bytes | None = None

@dataclass
class FileEditEmission:
    path: str
    search: str
    operation: Literal["replace", "insert-after", "insert-before"]
    content: str
    match_all: bool = False
    title: str | None = None
    signature: bytes | None = None

Emission = (TextEmission | ThinkingEmission | PythonEmission
            | TerminalEmission | FileWriteEmission | FileEditEmission)
```

### ActionEvent reshape

```python
# agex/agent/events.py

@dataclass
class ActionEvent(Event):
    agent_name: str
    emissions: list[Emission]
```

Gone: `title`, `thinking`, `report`, `code`, `terminal`,
`file_actions`, `terminal_action`.

### Part-level emission_id

```python
# agex/eval/objects.py

@dataclass
class PrintAction:
    args: tuple
    emission_id: str | None = None

@dataclass
class ImageAction:
    image: Any
    emission_id: str | None = None
```

Old `PrintAction` was tuple-shaped; any code iterating it directly
(`" ".join(str(a) for a in pa)`) needs to iterate `pa.args`.

### TokenChunk extension

```python
# agex/llm/core.py

@dataclass
class TokenChunk:
    type: Literal[..., "text"]   # add "text" for TextEmission
    content: str
    done: bool
    emission_index: int = 0       # new
    input_tokens: int | None = None
    output_tokens: int | None = None
```

### Drop

- `FileAction`, `EditAction` from `agex/agent/datatypes.py` — superseded by `FileWriteEmission` / `FileEditEmission`
- `task_continue` builtin from `agex/eval/builtins/task_control.py` (or wherever it lives)
- `TaskContinue` exception class in `agex/eval/bridge/result.py`

### Checkpoint

```bash
python -c "from agex.agent.emissions import *"   # imports clean
python -c "from agex.agent.events import ActionEvent; a = ActionEvent(agent_name='x', emissions=[])"
```

Tests still broken at this point — nothing's wired up. Don't run the
full suite yet.

### Scope

Small. ~2 hours.

---

## Phase 2 — Renderer + execution loop

**Goal:** get a turn end-to-end through the new shape. Hard-wire
tool-use format, drop task_continue from the loop semantics.

### Renderer (`agex/llm/formats/tool_use/renderer.py`)

Rewrite `render_events_as_tool_use` to walk `event.emissions` in
order. Each emission type maps to a block:

| Emission | Block |
|---|---|
| `TextEmission` | `{"type": "text", "text": ...}` in assistant content |
| `ThinkingEmission` | `{"type": "thinking", ...}` (Anthropic) or `{"type": "text"}` narration fallback |
| `PythonEmission` | `{"type": "tool_use", "name": "python_action", ...}` |
| `TerminalEmission` | `{"type": "tool_use", "name": "terminal_action", ...}` |
| `FileWriteEmission` | `{"type": "tool_use", "name": "write_file", ...}` |
| `FileEditEmission` | `{"type": "tool_use", "name": "edit_file", ...}` |

Each tool-call emission gets a stable `id` (e.g. `toolu_{task}_{event_idx}_{emission_idx}`).

Observation pairing: each tool-call emission needs a `tool_result` in
the next user message. OutputEvent parts with `emission_id=X` pair
back to emission X's tool_use block.

### Execution loop (`agex/agent/loop/`)

Replace single-main-action logic with sequential emission walk:

```python
observations_by_emission_id: dict[str, list] = {}
terminator = None
for emission in action_event.emissions:
    if isinstance(emission, (TextEmission, ThinkingEmission)):
        continue  # logged, not executed
    if isinstance(emission, (FileWriteEmission, FileEditEmission)):
        apply_to_fs(emission)
        observations_by_emission_id[emission.id] = synthesize_result(emission)
    elif isinstance(emission, PythonEmission):
        result = execute_in_sandbox(emission.code, namespace=shared_ns)
        observations_by_emission_id[emission.id] = gather_outputs(result, emission.id)
        if result.terminator:
            terminator = result.terminator
            break
    elif isinstance(emission, TerminalEmission):
        result = execute_shell(emission.commands)
        observations_by_emission_id[emission.id] = [PrintAction(args=(result.output,), emission_id=emission.id)]
```

Key points:
- **Shared namespace** across PythonEmissions in one turn
- **Early break** on first terminator; remaining emissions stay in the log but don't execute
- **No implicit "continue"** — Python just returning is the happy path (no exception raised)
- **OutputEvents tagged** with their source `emission_id`

### Drop task_continue

- Remove the builtin from the sandbox namespace
- Remove `TaskContinue` exception
- Update `handle_result` in `agex/eval/bridge/result.py` — the section that currently converts `TaskContinue.observations` → OutputEvent can go
- Python completing without a terminator = turn continues, loop proceeds to next LLM call

### Checkpoint

```bash
python -m pytest tests/agex/llm/test_tool_use_renderer.py -x
```

Renderer tests should pass (they'll need updates for the new event
shape but the shape should be testable in isolation). A small
integration test that drives one mock turn end-to-end:

```python
# tests/agex/agent/test_multi_emission_turn.py
def test_three_emissions_in_one_turn():
    event = ActionEvent(agent_name="a", emissions=[
        FileWriteEmission(path="/foo.py", content="X=1"),
        PythonEmission(code="from foo import X; print(X)"),
        TextEmission(text="done!"),
    ])
    # assert renderer produces 3 tool_use blocks + 1 text block in order
    # assert execution runs file write, then python, with observations
    # paired to emission[1]
```

### Scope

Medium. ~1 day. The execution loop is the fiddliest piece.

---

## Phase 3 — Wire format

**Goal:** Native thinking support at the schema level, primer
slim-down, XML deletion.

### ToolUseWireFormat changes

```python
class ToolUseWireFormat:
    def __init__(self, native_thinking: bool = False):
        self.native_thinking = native_thinking

    def tool_schema(self) -> list[dict]:
        # When native_thinking=True, drop the `thinking` parameter
        # from python_action and terminal_action schemas.
        ...

    def format_primer(self) -> str:
        # When native_thinking=True, drop the THINKING narration
        # section. Also drop the task_continue section unconditionally.
        ...
```

Additional primer slims:
- No task_continue section
- No "wrap your user-facing update in report" section  
- No "emit exactly one python_action per turn" discipline

### Delete XML

Remove:
- `agex/llm/formats/xml/` (entire directory)
- XML import/re-export from `agex/llm/formats/__init__.py`
- XML-mode tests (`tests/agex/llm/test_xml_*.py` if they exist separately; else gut the XML branches from existing tests)
- XML-mode branches in every client's `complete_stream` / `acomplete_stream`
- XML-mode primer content in `agex/agent/primer_text.py`

Keep:
- `agex/llm/formats/wire_format.py` — the `WireFormat` protocol
- `wire_format` kwarg on clients (always a `ToolUseWireFormat` now)

### Checkpoint

```bash
python -m pytest tests/agex/llm/ -q
```

LLM-layer tests pass with either `native_thinking=True` or `False`
depending on fixture. XML-mode tests deleted.

### Scope

Small-medium. ~4 hours.

---

## Phase 4 — Provider adapters

**Goal:** signature round-trip working for Gemini 3 and Claude 4.6.
OpenAI gets a sensible reasoning-effort default.

### Gemini (`agex/llm/formats/tool_use/gemini_adapter.py` + `gemini_client.py`)

**Stream translator:** capture `thought_signature` on each
`function_call` part. Emit via a new event type or extend
`ToolCallEnd` with a `signature: bytes | None` field. Store it on the
corresponding emission as it's built.

**Message translator:** when rendering history, read each emission's
`signature` and include it on the `function_call` part:

```python
{
    "function_call": {
        "id": emission.id,
        "name": "python_action",
        "args": {...},
        "thought_signature": emission.signature,  # new
    }
}
```

**Client:** set `native_thinking=True` on the default wire format.
Pass `include_thoughts=True` in thinking_config so we can capture
thinking content text for ThinkingEmission.

### Anthropic (`agex/llm/formats/tool_use/anthropic_adapter.py` + `anthropic_client.py`)

**Stream translator:** capture thinking-block content + signatures
as they arrive. Emit ThinkingEmissions in order relative to
tool_uses. Preserve `redacted_thinking` blocks as
`ThinkingEmission(text="", signature=..., redacted=True)`.

**Message translator:** when rendering history, emit thinking blocks
(with signatures) from ThinkingEmissions in their original position
relative to tool_uses. Claude's interleaved thinking is
structurally preserved by the emission list order.

**Client:** set `native_thinking=True` on the default wire format.
Switch to adaptive thinking API:

```python
thinking={"type": "adaptive", "effort": "low"}  # sensible default
```

### OpenAI (`agex/llm/openai_client.py` + `pyfetch_openai.py`)

**No signature work** — reasoning is server-side, nothing to
round-trip on Chat Completions.

**Set `native_thinking=True`** on the default wire format. The
primer/schema stops asking the model to narrate since GPT-5+ is
thinking natively.

**Sensible default:**

```python
# If caller doesn't override, agex defaults reasoning.effort="low"
# so GPT-5.x actually reasons during agentic work.
request_kwargs.setdefault("reasoning", {"effort": "low"})
```

Document that callers can override to `"none"`, `"minimal"`,
`"medium"`, `"high"`.

### Checkpoint

Integration tests against mock streams for each provider:

```bash
python -m pytest tests/agex/llm/test_gemini_signature_roundtrip.py
python -m pytest tests/agex/llm/test_anthropic_interleaved_thinking.py
python -m pytest tests/agex/llm/test_openai_reasoning_default.py
```

Plus re-run the failing `examples/funcy.py` against Gemini 3 — it
should complete two prompts in a row without the 400.

### Scope

Large. ~2 days. Gemini signature round-trip is the main work;
Claude interleaved is subtler but doable; OpenAI is a one-liner.

---

## Phase 5 — Cleanup sweep

**Goal:** delete dead code, update docs, land on green main.

### Dead code to remove

- Any remaining references to old `FileAction` / `EditAction`
- Any `task_continue` documentation or examples
- XML-mode references in CHANGELOG / docs / READMEs
- Old tests that can't be adapted (don't force-fit — rewrite or delete)

### Docs to write

- `docs/concepts/reasoning.md` — per-provider reasoning-effort patterns
- `docs/concepts/wire-format.md` — short page explaining
  `ToolUseWireFormat` and `native_thinking`
- CHANGELOG entry for the breaking reshape
- Update `docs/quick-start.md` if any examples use dropped APIs
- Update `retooling.md` status to "implemented" (link to the commit)

### Examples

- `examples/funcy.py` — re-verify end-to-end
- Any other example that used `task_continue` — rewrite to rely on
  implicit continue
- Remove `report=` usage from examples

### Final checkpoint

```bash
python -m pytest -q    # full suite
python examples/funcy.py
python examples/<any agent-chat example>
```

Version bump: `0.11.0` (breaking reshape warrants a minor bump;
we're pre-1.0 so this is within convention).

### Scope

Small-medium. ~4 hours.

---

## Phase 6 — agex-studio follow-up

**Goal:** studio consumes the new wheel cleanly.

### Studio code to update (`agex-studio/src/lib/`)

- `agent.js` — system prompt references (drop task_continue guidance)
- `pyodide.js` `handleToken` — route by `emission_index`, handle
  new `text` TokenChunk type
- `sessions.js` `loadHistory` — read new ActionEvent shape, emit
  per-emission UI blocks in order
- `chat.js` / accordion rendering — multi-emission-per-turn layout

### Studio test updates

- Event-log snapshot fixtures need regeneration
- Token-stream fixtures need `emission_index` added
- Settings test: drop `toolUseWireFormat` field entirely if we're
  removing the toggle (no XML mode = no toggle needed)

### Scope

Medium. ~1 day. Mostly mechanical once the new shape is stable.

---

## Risks and mitigations

1. **Claude interleaved thinking is under-tested.** Our mock streams
   may miss real-world quirks. Mitigation: run a real end-to-end
   conversation against Sonnet 4.6 with `thinking.adaptive` before
   declaring Phase 4 done.
2. **Gemini signature encoding.** `thought_signature` is bytes.
   Event-log persistence needs to handle bytes correctly (base64 for
   JSON, native for pickle). Mitigation: add a round-trip test that
   persists and reloads an ActionEvent with signed emissions.
3. **Parallel function_calls on Gemini 3 / OpenAI tool_calls.** The
   SDK may stream multiple tool_calls with the same `index` or
   overlapping timing. The translator needs to handle this cleanly.
   Mitigation: explicit multi-tool-call test fixtures per provider.
4. **Renderer/execution mismatch.** The renderer must build tool_use
   IDs the execution loop can pair observations back to. Bug here
   would cause tool_result orphans. Mitigation: invariant check in
   the renderer — every tool-call emission produces exactly one
   tool_use block and exactly one tool_result pairing.

## Non-goals

- **OpenAI Responses API migration.** Would enable interleaved text
  on OpenAI but is a bigger shift. Defer.
- **Parallel tool execution.** We execute sequentially regardless of
  how the provider presented parallelism. Simpler mental model,
  Python-namespace safe.
- **Event log schema migration.** Clean break — old logs won't load.
  Acceptable given sole-user status.
