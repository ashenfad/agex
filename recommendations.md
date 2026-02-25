# Agex Recommendations

## High Priority

### 1. Surface LLM Token Usage on ActionEvents
All three providers (Anthropic, OpenAI, Gemini) return actual token usage in their responses, but the streaming methods consume the stream and discard this data. Add `input_tokens: int | None` and `output_tokens: int | None` fields to `LLMResponse` and propagate them to `ActionEvent`. Each provider's stream handler should capture usage from the final message/chunk:

- **Anthropic**: `get_final_message().usage.input_tokens` / `.output_tokens` (plus cache read/creation tokens)
- **OpenAI**: Final stream chunk with `stream_options={"include_usage": True}` → `usage.prompt_tokens` / `.completion_tokens`
- **Gemini**: `response.usage_metadata.prompt_token_count` / `.candidates_token_count`

This is the single missing piece for cost tracking via `on_event`. The existing `full_detail_tokens` / `low_detail_tokens` on `BaseEvent` are tiktoken estimates for context budgeting — not actual API usage. With real usage numbers on events, an `on_event` handler can trivially compute cost, track cumulative spend, or aggregate across runs.

### 2. Add Retry-with-Backoff for LLM API Calls
LLM API calls are the most common production failure point — transient 429 (rate limit) or 503 (overloaded) errors from Anthropic/OpenAI/Gemini currently surface as a `RuntimeError` and fail the turn. Adding `tenacity` as a dependency and wrapping API calls with retry + exponential backoff would handle the most frequent production failure mode automatically.

## Low Priority

### 3. Remove Unused xxhash Dependency
`xxhash` is listed in `pyproject.toml` as a runtime dependency but has no imports anywhere in the agex source. Remove it.

### ~~4. Replace psutil with resource.getrusage~~
Done — memory limits are now handled entirely by sandtrap's `Policy.memory_limit`. `psutil` removed.
