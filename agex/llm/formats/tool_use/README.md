# `tool_use` wire format

Implementation reference for the one wire format agex uses to talk
to every provider.  See `agex/llm/README.md` for per-provider
reasoning round-trip.

## The shape of a turn

Every agent turn is an **ordered list of emissions** — one
`ActionEvent(emissions=[...])` in the event log.  Each emission maps
cleanly to a native provider content block, so agex stays a thin
translation layer rather than teaching the model a dialect.

| Emission | Native block on the wire |
|---|---|
| `PythonEmission(code, title?, thinking?)` | `tool_use` / `function_call` — `python_action` |
| `TerminalEmission(commands, title?, thinking?)` | `tool_use` / `function_call` — `terminal_action` |
| `FileWriteEmission(path, content, mode)` | `tool_use` / `function_call` — `write_file` |
| `FileEditEmission(path, search, content, match_all)` | `tool_use` / `function_call` — `edit_file` |
| `TextEmission(text)` | assistant `text` block |
| `ThinkingEmission(text, signature?, redacted?)` | provider thinking block (Claude `thinking`, Gemini thought part, OpenAI Responses reasoning, OpenRouter `reasoning_details`) |

A single turn can carry several emissions in any order — e.g.
`[ThinkingEmission, FileWriteEmission, PythonEmission]`.  The loop
walks them in emission order: file writes hit the VFS before the
subsequent `python_action` runs, so you can write a helper module and
import it the same turn.  PythonEmissions share a namespace — later
ones see variables bound earlier.  Returning normally from Python is
the implicit continue; explicit control uses `task_success(result)` /
`task_fail(msg)` / `task_clarify(msg)` inside
`python_action.code`.

## Tools are the interface

agex configures every provider to force a tool call each turn
(`tool_choice="required"` on OpenAI/OpenRouter, `{"type": "any"}` on
Anthropic — skipped when extended thinking is on, since the API
rejects the combo — `tool_config.mode="ANY"` on Gemini).  Plain
assistant text isn't a reply channel: it doesn't execute anything,
doesn't finish the task.  Questions go through `task_clarify`; status
updates go through `print(...)` in `python_action` so they surface in
the next `tool_result`.

Only `python_action` and `terminal_action` produce observations
(stdout, errors, shell output) that the model sees next turn.
`write_file` / `edit_file` produce only a synthesized `✓ wrote /path`
confirmation — when you want feedback on a file you just wrote, pair
it with a `python_action` that imports or tests it in the same turn.

## `ToolUseWireFormat` and `native_thinking`

`ToolUseWireFormat(native_thinking=True|False)` is the single wire
format.  The flag controls whether **thinking rides in the tool
schema** or in **native content blocks**:

- `native_thinking=True` (default for all providers) — `python_action` /
  `terminal_action` drop their `thinking` parameter.  Reasoning
  arrives in provider-native thinking blocks instead: Claude
  extended-thinking, Gemini thought parts with signatures, OpenAI
  Responses reasoning items, OpenRouter `reasoning_details`.  Every
  block's signature round-trips through `ThinkingEmission.signature`
  so the model's prior reasoning is preserved byte-for-byte on
  subsequent turns.
- `native_thinking=False` — `python_action` / `terminal_action`
  carry a `thinking` parameter that the model fills with narration
  before its code.  Safe fallback for older models (Claude 3.x,
  non-reasoning OpenAI, anything on OpenRouter that doesn't forward
  reasoning).

Users on a non-reasoning model can opt out per-client:

```python
from agex.llm import connect_llm
from agex.llm.formats import ToolUseWireFormat

llm = connect_llm(
    provider="anthropic",
    model="claude-3-5-haiku-latest",
    wire_format=ToolUseWireFormat(native_thinking=False),
)
```

## Primer

The primer is the system-prompt addendum that teaches the model the
handful of semantics JSON Schema can't express on its own:

- Tools are the entire interface; every turn must call one.
- `task_success` / `task_fail` / `task_clarify` live **inside**
  `python_action.code`, not as separate tools.
- File tools run before subsequent `python_action` within a turn;
  write a helper and import it the same turn.
- `write_file` with `mode="append"` over `edit_file` when adding new
  content to an existing file — append can't miss a search target
  that was never there.
- `edit_file` does one thing: swap `search` for `replace`.  To insert
  around an anchor, include the anchor in `replace`.
- Observations next turn only come from `python_action` /
  `terminal_action`; pair file writes with a test.

The primer is slimmer on `native_thinking=True` — the
narration-in-schema instructions are moot because the `thinking`
parameter is gone from the schema.

## What got deleted

The old XML wire format (`<THINKING>` / `<CODE>` tags parsed from
text streams) is gone, along with `task_continue` (normal return is
the implicit continue) and the single-`python_action`-per-turn
restriction.  The net effect: agex stops asking the model to learn a
dialect, and model tool-use fluency improves because the wire shape
matches the provider's training distribution.
