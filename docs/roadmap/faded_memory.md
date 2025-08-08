# Faded Memory (Dynamic Context Management)

Re-render older observations at lower detail with deterministic, local decay policies. Allow agents to “refresh” details on demand from ground-truth state.

## Concept
- Apply exponential/linear decay to token budgets per observation as turns grow.
- Re-render from Versioned state snapshots; no LLM summarization needed.
- Allow active refresh (e.g., printing a variable) to bring details forward.

## Example Policy (exponential)
- Most recent: full budget
- Previous: 1/2 budget
- Earlier: 1/5 budget

## Benefits
- Long-running conversations with graceful degradation.
- High fidelity by re-rendering from source state.
- Cheap and deterministic vs. LLM summaries.

## Considerations
- Integrate with system token counting and visibility.
- UX for refresh hooks; caching for performance.

Related issue: [Issue #5](https://github.com/ashenfad/agex/issues/5)
