# MIME Bundles for Events

Serialize events (especially OutputEvent parts) into JSON-safe MIME bundles for UIs, logging, and replay.

## Concept
- `render(event) -> dict`: canonical, JSON-serializable structure for all events.
- Rich parts converted via `_repr_mimebundle_`, `_repr_html_`, `_repr_png_`, with fallbacks to `text/plain`.

## Example
```json
{
  "event_type": "OutputEvent",
  "agent_name": "data_analyst",
  "parts": [
    {"text/plain": "'Here is the dataframe:'"},
    {"text/plain": "A B...", "text/html": "<div>...</div>"}
  ]
}
```

## Benefits
- Web/UI friendly, observability ready.
- Standardizes multi-representation payloads.
- Graceful degradation with fallbacks.

## Considerations
- Truncation and size limits to protect transports.
- Sanitize/escape HTML; base64 encoding for binaries.
- Single implementation used by streamers and hooks.

Related issue: [Issue #3](https://github.com/ashenfad/agex/issues/3)
