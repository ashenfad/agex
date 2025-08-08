# Contextual, Capability-Driven Primers

Auto-inject small, targeted coaching based on registered capabilities. The system assembles guidance snippets just-in-time so the agent learns how to use relevant tools without polluting context.

## Concept
- Triggers fire on registration (e.g., `agent.module(conn, name="db")`).
- A primer assembler deduplicates, orders, and budgets snippets before each LLM call.
- Snippets are small, composable fragments (e.g., safe patterns for unpicklables, `view_image` usage).

## Benefits
- Reduces boilerplate primers like `db_primer.py`.
- Keeps context lean; injects only when capabilities are present.
- Teaches best practices at the moment of need.

## Considerations
- Budget control integrates with system token counting.
- User-extensible primer library with clear triggers (module/class/instance patterns).

Related issue: [Issue #1](https://github.com/ashenfad/agex/issues/1)
