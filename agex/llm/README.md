# `agex.llm` — LLM clients and reasoning round-trip

Implementation reference for how each provider client captures,
round-trips, and replays native model reasoning across turns.  See
`agex/llm/formats/tool_use/README.md` for the wire format itself
(emission list, tool surface, primer).

## Why it matters

Modern frontier models (Claude 4+, Gemini 3, GPT-5+, Grok, DeepSeek)
emit **native thinking blocks** alongside tool calls.  The provider
signs this state — usually an opaque bytes payload plus an id — and
requires it to be **replayed verbatim** on subsequent turns.  Drop
the signature and Gemini 3 returns a 400; drop it on Claude and the
next turn starts reasoning from scratch.  agex captures every
signature off the wire, stores it on
`ThinkingEmission.signature`, and stitches it back into the right
shape on the next request.

## Per-provider model

### OpenAI — Responses API

For `gpt-5*` / `o1*` / `o3*` models, `Agent(llm=connect_llm(provider="openai"))`
routes to `/v1/responses` rather than Chat Completions.  Chat
Completions rejects `reasoning_effort` combined with function tools
for several GPT-5-family models; Responses is the sanctioned endpoint
and is where new features ship (encrypted reasoning round-trip,
typed output-item events).

Wire:

- Request: `reasoning={"effort": "low"}` (fold from legacy
  `reasoning_effort="low"` if set), `store=False`,
  `include=["reasoning.encrypted_content"]`.
- Response streams `response.output_item.added` / `...done` events
  for `type: "reasoning"` items carrying an `id` and
  `encrypted_content`.  We pack both into a
  `ThinkingEmission.signature` with an `openai-responses:` tag
  prefix; on replay, `translate_messages_to_openai_responses` unpacks
  the signature back into a `reasoning` input item at the same
  position.

Non-reasoning OpenAI models (gpt-4o etc.) stay on Chat Completions
and receive no `reasoning` kwarg — the endpoint dispatches
automatically by model name, with `use_responses=` as an explicit
override.

### Anthropic — extended thinking

Claude 4+ supports **extended thinking** natively.  The Anthropic
client defaults to `native_thinking=True` and injects
`thinking={"type": "enabled", "budget_tokens": 2048}` into the
request kwargs (override by passing your own `thinking=`, including
`thinking=None` for Claude 3.x models that don't support it).

Wire:

- Streamed as typed content blocks: `thinking` (plain), `redacted_thinking`
  (opaque).  `_ThinkingState` accumulates `thinking_delta` +
  `signature_delta` events and flushes a single `ThinkingPart` per
  block.  Claude signs thinking blocks with a string; we encode it
  to bytes on the way in and back on the way out so
  `ThinkingEmission.signature` stays a provider-neutral `bytes | None`.
- Interleaved thinking round-trips by position — a turn that emitted
  `[thinking, tool_use, thinking, tool_use]` replays that exact
  sequence.

Extended thinking is incompatible with `tool_choice={"type": "any"}`;
agex's `_ensure_tool_choice_any` helper auto-skips the force when
`thinking` is in the request kwargs.  The no-progress nudge (see
below) is the soft substitute.

### Gemini — thought parts

Gemini 3 emits **thought parts** with `thought_signature` bytes on the
same `Part` as their `function_call`, or on a separate sibling part
that signs reasoning preceding subsequent calls.  The Gemini client
sets `thinking_config=ThinkingConfig(include_thoughts=True)` and
`tool_config.function_calling_config.mode="ANY"`.

Wire:

- The stream walker buffers every `candidates[].content.parts[]`
  across chunks — signatures sometimes arrive on a later chunk than
  the function_call they sign, so we flush at stream end with the
  latest-seen signature.  Thought parts get their own emission_index
  so replay position matches the original turn.
- On replay, `translate_messages_to_gemini` puts `thought_signature`
  as a sibling of the `function_call` (not inside it — the SDK's
  pydantic model rejects the extra field).
- Fallback: Gemini 3 occasionally returns a first `function_call`
  with no signature, but then 400s on the next turn demanding one.
  The docs sanction a literal dummy signature
  (`b"context_engineering_is_the_way_to_go"`) that skips validation;
  we inject it on the first `function_call` of any turn whose
  captured signature is None.

### OpenRouter — unified `reasoning_details`

OpenRouter's [unified reasoning-tokens API](https://openrouter.ai/docs/use-cases/reasoning-tokens)
normalizes every upstream provider to a single shape:

- Request: `reasoning={"enabled": True, "effort": "low"}`.
  `PyfetchOpenAI` injects this on the Chat Completions path whenever
  the wire format is native.
- Response: `choices[].delta.reasoning_details[]` entries with `type`
  (`reasoning.summary` / `reasoning.text` / `reasoning.encrypted`),
  `format` (`anthropic-claude-v1`, `openai-responses-v1`,
  `google-gemini-v1`, ...), `id`, `index`, and content.  The adapter
  accumulates by `index` and packs the full array into
  `ThinkingEmission.signature` with an `openrouter-reasoning:` tag
  prefix.  On replay, `translate_messages_to_openai` decodes the
  signature and attaches `reasoning_details` back to the assistant
  message — byte-for-byte identical to what the server sent.

This is what lets `PyfetchOpenAI` → OpenRouter → Claude / Gemini /
DeepSeek get the same reasoning round-trip as the direct clients
without knowing which upstream model is actually serving the request.

## Signature packing

All four providers end up on the same `ThinkingEmission.signature:
bytes | None` field, distinguished by a short tag prefix so each
adapter fails-closed on anything that isn't its own:

| Provider | Prefix | Payload |
|---|---|---|
| OpenAI Responses | `openai-responses:` | `{"id": rs_id, "encrypted_content": ...}` |
| OpenRouter reasoning | `openrouter-reasoning:` | full `reasoning_details` array |
| Anthropic extended thinking | *(no tag — raw signature bytes)* | Claude's opaque base64 string, utf-8 encoded |
| Gemini thought parts | *(no tag — raw signature bytes)* | Gemini's `thought_signature` bytes |

A Gemini signature handed to the OpenAI-Responses adapter decodes as
`None` and gets silently dropped rather than mis-replayed as an
id-less reasoning item.

## Nudges

Two guidance events fill in for the `task_continue` contract the
retooling removed:

- **Silent-python nudge** — fires when a `python_action` ran but
  produced no OutputEvent (no print, no view_image, no error).  The
  model sees *"your code executed but produced no observation; call
  task_success(...) to finish or keep going"* on the next turn.
- **No-progress nudge** — fires when a turn contained only
  `TextEmission` / `ThinkingEmission`, no actionable tool call.  The
  model sees *"No tools called last turn — plain text doesn't execute
  anything."*  Particularly useful on providers where
  `tool_choice={"type": "any"}` isn't available (e.g., Anthropic
  with extended thinking).

Both nudges pair cleanly with forced tool choice on providers that
support it, and fill the gap on those that don't.

## Opting out

Every provider's `wire_format=` and reasoning kwargs are overridable:

```python
from agex.llm import connect_llm
from agex.llm.formats import ToolUseWireFormat

# Plain chat-class Claude — drop extended thinking, narrate in schema.
llm = connect_llm(
    provider="anthropic",
    model="claude-3-5-haiku-latest",
    wire_format=ToolUseWireFormat(native_thinking=False),
    thinking=None,
)

# Force a GPT-5 model onto Chat Completions.
llm = connect_llm(provider="openai", model="gpt-5-mini", use_responses=False)

# Custom OpenRouter reasoning budget.
from agex.llm.pyfetch_openai import PyfetchOpenAI
llm = PyfetchOpenAI(
    model="anthropic/claude-sonnet-4.5",
    api_key=...,
    reasoning={"effort": "high", "max_tokens": 4000},
)
```
