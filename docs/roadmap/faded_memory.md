# Faded Memory (Dynamic Context Management)

**Status: ✅ Implemented** (agent-directed chaptering)

Manage growing agent context by letting agents close completed work into named chapters — lossless, browsable, and agent-controlled.

## What's Implemented

agex uses **agent-directed chaptering** for context management:

- When `input_tokens` exceeds `log_high_water_tokens`, the framework triggers a `__chapter__` task
- The agent reviews its event history and creates `Chapter` instances to close out completed work
- Original events are preserved inside `ChapterEvent` and browsable via a `/chapters` VFS overlay
- Chaptering runs up to 3 rounds, stopping when `input_tokens` drops below `log_low_water_tokens`

This replaced the earlier automated summarization approach (SummaryEvent + low-detail rendering tiers), which was removed in favor of giving agents direct control over what gets compacted and how.

### Benefits

✅ **Agent autonomy**: The agent decides what to chapter and writes its own summaries
✅ **Lossless**: Original events preserved inside chapters and browsable via VFS
✅ **Natural boundaries**: Chapters align with logical phases of work, not arbitrary token counts
✅ **Long-running agents**: Context stays manageable across many tasks and iterations

## Future Explorations

- **Low-detail rendering tiers**: Re-introduce deterministic budget reduction for older-but-unchaptered events (e.g., images → placeholders, truncated output). This would complement chaptering by reducing token cost of events the agent hasn't chaptered yet.

- **Cache-aware chaptering**: Coordinate chapter boundaries with provider-side cache breakpoints (e.g., Anthropic's cache markers) to maximize cache hits when context is rebuilt.

## Documentation

- [Concepts: Chapters](../concepts/chapters.md)

Related issue: [Issue #5](https://github.com/ashenfad/agex/issues/5)
