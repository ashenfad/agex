# Faded Memory (Dynamic Context Management)

**Status: ✅ Implemented** (3-tier approach with low-detail rendering + summarization)

Re-render older observations at lower detail with deterministic, local decay policies, combined with LLM summarization for oldest events.

## What's Implemented

agex now uses a **3-tier context management system**:

1. **Full Detail** (events newer than threshold): Complete rendering with full images, deep nesting (depth 4), verbose output
2. **Low Detail** (events older than threshold): Deterministic compression via reduced rendering budgets
   - Images → `[Image]` text placeholders (~1000 tokens saved per image)
   - Nested structures: depth 4 → depth 2
   - List items: 25 → 10 items
   - Uses ~25-50% of full detail tokens
3. **Summarized** (oldest): LLM-generated text summary replacing multiple old events

### How It Works

- When event log exceeds `log_high_water_tokens`, summarization triggers
- Creates a `SummaryEvent` with a **fixed** `low_detail_threshold` timestamp (initially at 25th percentile by age)
- Events older than threshold automatically render at low detail in future agent context
- As new events arrive between summarizations, they accumulate in the "full detail" bucket (newer than fixed threshold)
- Uses correct token counts when deciding what to keep (low for old, full for new)
- Cache-friendly: threshold is stored in summary, individual events remain immutable

### Benefits Realized

✅ **Long-running agents**: Significantly more event history kept in context  
✅ **Graceful degradation**: Tiered approach preserves structure better than pure summarization  
✅ **Deterministic + cheap**: Low-detail rendering is fast and predictable  
✅ **Token efficiency**: Low-detail events use 25-50% of tokens, allowing 2-4x more events  
✅ **Cache preservation**: Summarization only happens when needed, individual events never mutate

## Future Explorations

- **Progressive threshold updates**: Between summarizations, periodically bump the low-detail threshold forward to keep the full-detail section small. This would maximize provider-side cache hits by keeping the cached portion (newest events) stable and compact. Requires careful cache-breakpoint management (e.g., Anthropic's cache markers at the start of full-detail section).
  
- **Adaptive thresholds**: Adjust 25/75 split based on event types (e.g., keep more full-detail for visual/interactive tasks, compress more aggressively for text-heavy analysis).
  
- **Continuous decay**: More granular budget tiers (current: 2-tier = full/low). Could introduce 3-4 budget levels with progressive compression as events age.

## Documentation

- [Agent: Event Log Summarization](../api/agent.md#event-log-summarization)
- [Events: SummaryEvent](../api/events.md#summaryevent)

Related issue: [Issue #5](https://github.com/ashenfad/agex/issues/5)
