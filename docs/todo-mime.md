# `agex.events.render()`: MIME Bundle Conversion for Rich Events

## The Goal

To provide a robust, standardized way to serialize `agex` event objects into JSON-safe dictionaries, with a primary focus on handling rich Python objects (like DataFrames, Figures, etc.) contained within `OutputEvent`s. This is critical for two key use cases:
1.  **Web UI Integration**: Streaming events to a remote UI requires a JSON-serializable format.
2.  **Production Observability**: Logging events to platforms like OpenTelemetry or Datadog requires a safe, structured representation.

## The Design: A Central `render()` Helper

The core of the solution is a single, public helper function: `agex.events.render(event: BaseEvent) -> dict`.

### Behavior

1.  **Input**: Takes any raw `agex` event object (e.g., `TaskStartEvent`, `ActionEvent`, `OutputEvent`).
2.  **Output**: Returns a JSON-serializable dictionary.
3.  **Standard Events**: For simple events without rich objects (like `TaskStartEvent` or `ActionEvent`), the function is equivalent to `event.model_dump()`.
4.  **Rich Events (`OutputEvent`)**: When the input is an `OutputEvent`, the `render()` function performs a special conversion on the `event.parts` list:
    *   It iterates through each object in `parts`.
    *   For each object, it generates a standard [MIME bundle](https://ipython.readthedocs.io/en/stable/config/integrating.html#mime-bundles), the same format used by Jupyter/IPython.
    *   A MIME bundle is a dictionary where keys are MIME types (e.g., `'text/plain'`, `'text/html'`, `'image/png'`) and values are the corresponding data representations.
    *   This conversion will intelligently use existing `_repr_mimebundle_`, `_repr_html_`, `_repr_png_`, etc., methods on the objects if they exist.
5.  **Structure**: The final dictionary for an `OutputEvent` will contain the standard event fields, but its `parts` list will be replaced by the list of generated MIME bundles.

### Example

**Input (Python objects):**
```python
# Raw event
output_event = OutputEvent(
    agent_name="data_analyst", 
    parts=[
        "Here is the dataframe:", 
        pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    ]
)
```

**Output of `render(output_event)` (JSON-serializable dict):**
```json
{
  "event_type": "OutputEvent",
  "agent_name": "data_analyst",
  "timestamp": "...",
  "parts": [
    {
      "text/plain": "'Here is the dataframe:'"
    },
    {
      "text/plain": "   A  B\\n0  1  3\\n1  2  4",
      "text/html": "<div>...html table...</div>",
      "application/json": {
          "schema": { "...
          },
          "data": [ ... ]
      }
    }
  ]
}
```

## Implementation Details

*   The `render` function should live in `agex/render/value.py` or a similar central rendering module.
*   It should gracefully handle objects that cannot be converted, providing a best-effort `text/plain` representation using `repr()`.
*   It should be the single source of truth for event serialization, used by both the `mime_stream()` helper (for web UIs) and any `on_event` observability hooks.

This design provides a powerful and flexible way to handle rich data, preserving the integrity of the information while ensuring compatibility with standard serialization formats. 