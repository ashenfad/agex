# Capabilities Primer

A capabilities primer is a concise, curated description of what an agent can do with its registered functions, classes, and modules. It replaces (or suppresses) the verbose policy-rendered listings in the system context with a shorter, guidance‑oriented text.

## Motivation

- Rendered registrations can be long and redundant (duplicate signatures/docstrings).
- A compact “primer” that describes patterns of usage helps models focus and saves tokens.
- Keep this distinct from the agent’s behavioral primer and the built‑in primer.

## Terminology

- **Built‑in primer**: Framework guidance always included first.
- **Capabilities primer**: Optional, curated text describing exposed capabilities.
- **Agent primer**: Optional, user‑provided behavioral instructions for the agent.

System message assembly:

1. built‑in primer
2. capabilities primer OR rendered registrations (fallback)
3. agent primer

## Proposed API

### Attribute on Agent

- `agent.capabilities_primer: str | None`
  - If set (non‑empty), used instead of rendered registrations.
  - If `None`, the default policy rendering is used.
  - If `""` (empty string), the section is suppressed entirely.
  - Acceptable at construction via `Agent(..., capabilities_primer=None)` and assignable later.

### Pure builder (no side effects)

- `summarize_capabilities(agent, token_budget: int = 800, llm_client=None, use_cache: bool = True) -> str`
  - Renders current registrations (honors visibility), asks an LLM to compress into a concise capabilities primer within `token_budget` (tiktoken estimates).
  - `llm_client` (optional): If provided, use it for summarization; else use `agent.llm_client`.
  - `use_cache`: Reads/writes a simple on‑disk cache keyed by the agent’s registration fingerprint, token budget, and summarizer model id.

### Caching (visible, simple)

- Location: project‑local `.agex/primer_cache/`
- Filename pattern: `{agent}-{fp8}-tb{budget}-m{model}.md`
- File header (first lines): agent name, fingerprint, token_budget, created_at, optional token counts.
- Invalidation: automatic via fingerprint change; force refresh with `use_cache=False`.

## Token Counting Alignment

- `view(agent, focus="tokens")` reflects the actual content sent:
  - If `capabilities_primer` is set, count that for the “registered resources” section.
  - Otherwise, count the policy‑rendered registrations.

## Summarization Rubric (LLM prompt guidelines)

- Deduplicate function/class docstrings and similar signatures.
- Prefer patterns and how‑to usage over raw enumerations.
- Emphasize high‑visibility items and explicitly configured dotted members.
- Include 1–2 canonical usage snippets per capability cluster when helpful; avoid long examples.
- Mention key constraints (e.g., Versioned state pickling patterns, no async, decorator syntax limits) only when relevant to exposed capabilities.
- Avoid hallucinating names; post‑filter against actual registrations.

## Examples

### Build and attach

```python
from agex import Agent

# After registering capabilities...
text = summarize_capabilities(agent, token_budget=1000)
agent.capabilities_primer = text
```

### Build and save manually

```python
from pathlib import Path

text = summarize_capabilities(agent, token_budget=1000)
Path("primer.md").write_text(text)
# Later
agent.capabilities_primer = Path("primer.md").read_text()
```

## Future Extensions

- Scope controls: `include=("functions","classes","modules")`, style presets ("primer" | "bullets").
- Provider‑aware token estimates inferred from `llm_client`.
- Optional utilities: `list_capability_primers()`, `clear_capability_primers()` to manage cache files.

This design keeps read (builder) and write (attribute) concerns separate, maintains transparency (simple cache files), and aligns token budgeting with what will actually be sent to the model.
