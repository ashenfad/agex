# Agent

The `Agent` class is the main entry point for creating AI agents in agex. Each agent manages its own set of registered capabilities (see [Registration](registration.md)) and can execute tasks (see [Task](task.md)) through a secure Python environment.

## Constructor

```python
Agent(
    primer: str | None = None,
    timeout_seconds: float = 5.0,
    max_iterations: int = 10,
    name: str | None = None,
    capabilities_primer: str | None = None,
    llm_client: LLMClient | None = None,
    llm_max_retries: int = 2,
    llm_retry_backoff: float = 0.25,
    log_high_water_tokens: int | None = None,
    log_low_water_tokens: int | None = None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primer` | `str | None` | `None` | Instructions that guide the agent's behavior and personality |
| `timeout_seconds` | `float` | `5.0` | Maximum time in seconds for task execution |
| `max_iterations` | `int` | `10` | Maximum number of think-act cycles per task |
| `name` | `str | None` | `None` | Unique identifier for the agent (auto-generated if not provided) |
| `capabilities_primer` | `str | None` | `None` | Optional curated text that replaces the default capabilities listing (rendered from registrations). If `None`, the agent renders capabilities from registrations; if empty string, the section is suppressed. |
| `llm_client` | `LLMClient | None` | `None` | An instantiated `LLMClient` for the agent to use. If `None`, a default client is created. |
| `llm_max_retries` | `int` | `2` | Number of times to retry a failed LLM completion before aborting with `LLMFail`. |
| `llm_retry_backoff` | `float` | `0.25` | Initial backoff (seconds) between retries. Backoff grows exponentially per attempt. |
| `log_high_water_tokens` | `int | None` | `None` | Trigger event log summarization when total tokens exceed this threshold. If `None`, no automatic summarization occurs. |
| `log_low_water_tokens` | `int | None` | `None` | Target token count after summarization. Defaults to 50% of `log_high_water_tokens` if not specified. Must be less than `log_high_water_tokens`. |

### Examples

```python
from agex import Agent, connect_llm, LLMClient

# Simple agent using the default LLM (dummy provider or from env vars)
agent = Agent(primer="You are a helpful assistant.")

# Agent configured with a specific, explicitly created client
llm_client = connect_llm(provider="openai", model="gpt-4.1-nano", temperature=0.1)
expert_agent = Agent(
    primer="You are an expert data analyst.",
    llm_client=llm_client
)
```

## LLM Configuration

An agent's connection to a Large Language Model is managed by an `LLMClient` instance. There are two primary ways to configure this.

### 1. Direct Instantiation (Recommended)

The clearest and most explicit method is to create an `LLMClient` instance using the top-level `connect_llm()` factory function and pass it directly to the `Agent`'s constructor. This makes dependencies obvious and is ideal for production code and testing.

```python
from agex import connect_llm, Agent
from agex.llm.dummy_client import DummyLLMClient

# For production, create a client for a specific provider
prod_client = connect_llm(provider="openai", model="gpt-4.1-nano")
prod_agent = Agent(llm_client=prod_client)

# For testing, you can inject a dummy client
test_client = DummyLLMClient()
test_agent = Agent(llm_client=test_client)
```

### 2. Default Client (via Environment Variables)

If you do not pass an `llm_client` to the `Agent` constructor, `agex` will automatically create a default one for you by calling `connect_llm()` with no arguments. This default client is configured using environment variables.

```bash
# Example: Configure agent via environment variables
export AGEX_LLM_PROVIDER="openai"
export AGEX_LLM_MODEL="gpt-4.1-nano"
export OPENAI_API_KEY="your-key-here"
```

### 3. Using OpenAI-Compatible Endpoints (e.g., Ollama)

You can connect `agex` to any model provider that offers an OpenAI-compatible API endpoint, such as a local [Ollama](https://ollama.com/) server. This is done by specifying `provider="openai"` and passing the correct arguments to `connect_llm`.

```python
# Example for connecting to a local Ollama server
local_client = connect_llm(
    provider="openai",
    model="qwen3-coder:30b",   # The specific model served by Ollama
    base_url="http://localhost:11434/v1",
    api_key="ollama",          # Placeholder key for local services
)

local_agent = Agent(llm_client=local_client)
```

> **Note on Model Compatibility:** `agex` relies on the model's ability to follow specific function-calling or "tool use" instructions. While many models are compatible, we have specifically tested and verified that the `qwen3` family of models works effectively when served via Ollama. Performance may vary with other models. We recommend `qwen3-coder:30b`.

### 4. Advanced: Client vs. Completion Arguments

The `connect_llm` function is designed to intelligently separate two types of arguments:
-   **Client Arguments**: Used to configure the connection to the LLM provider (e.g., `api_key`, `base_url`, `timeout`).
-   **Completion Arguments**: Used to control the behavior of the model for each request (e.g., `temperature`, `top_p`, `max_tokens`).

You can pass both types of arguments directly to `connect_llm`. The underlying client for each provider (`OpenAIClient`, `AnthropicClient`, etc.) is responsible for correctly routing them.

```python
# Example with both client and completion arguments
client = connect_llm(
    provider="openai",
    model="gpt-4.1-nano",
    # --- Client Arguments ---
    api_key="sk-...",
    timeout=30.0,
    # --- Completion Arguments ---
    temperature=0.7,
    top_p=0.9,
)
```


## Properties

### `.name`
**Type:** `str`

The agent's unique identifier. If not provided in constructor, a random name is generated.

```python
agent = Agent()
print(agent.name)  # "agent_abc123" (auto-generated)

named_agent = Agent(name="my_assistant")
print(named_agent.name)  # "my_assistant"
```

### `.primer`
**Type:** `str | None`

The agent's behavioral instructions.

### `.capabilities_primer`

By default, the agent’s system message includes a capabilities section rendered from your registrations (functions, classes, modules), honoring their visibility levels. You can override this with a curated primer string.

Behavior:

- If `capabilities_primer` is `None` (default): render from registrations.
- If `capabilities_primer` is a non-empty string: use that text instead.

This lets you replace verbose listings with a concise, guidance-oriented document.

See the [Capabilities Primer Helper](#capabilities-primer-helper) section for how to generate these documents.


```python
agent = Agent(primer="You are concise and direct.")
print(agent.primer)  # "You are concise and direct."
```

### `.timeout_seconds`
**Type:** `float`

The maximum time in seconds allowed for a single block of agent-generated code to execute. This is a safety mechanism to prevent runaway code. It applies strictly to code execution time, not time spent waiting for the LLM.

### `.max_iterations`
**Type:** `int`

Maximum number of think-act cycles per task. If an agent doesn't complete a task within this limit, it will raise a `TaskTimeout`.


## Capabilities Primer Helper

Use the helper to generate a curated capabilities primer (markdown) from the agent's current registrations. Attach the result to `agent.capabilities_primer` (or pass via constructor) to replace the default visibility-based listing.

Signature:

```python
from agex import summarize_capabilities

def summarize_capabilities(
    agent: Agent,
    target_chars: int,
    llm_client: LLMClient | None = None,
    use_cache: bool = True,
) -> str: ...
```

Parameters:

- `agent`: The agent whose registered capabilities will be summarized (visibility-aware).
- `target_chars`: Minimum character count to target; the helper asks the model to write at least this many characters.
- `llm_client`: Optional override client for summarization (defaults to `agent.llm_client`).
- `use_cache`: If True, read/write a project-local cache under `.agex/primer_cache/` keyed by the agent fingerprint, target length, and model id.

Behavior:

- Renders the agent's capabilities per current registrations and visibility, then asks the model to synthesize a concise, guidance-oriented primer.
- Caches the result at a path like: `.agex/primer_cache/{agent}-{fp8}-ch{target_chars}-m{model}.md` with a small header (agent, fingerprint, target_chars, model, timestamp).
- The tokens-focused view (`view(agent, focus="tokens")`) counts the capabilities primer when present; otherwise it counts the rendered registrations.

Usage:

```python
text = summarize_capabilities(agent, target_chars=8000)
agent.capabilities_primer = text

# Or pass via constructor
# agent = Agent(capabilities_primer=text)

# Verify token budget with the primer applied
from agex import view
print(view(agent, focus="tokens"))
```

## Event Log Summarization

For long-running agents, the event log can grow large and consume significant context window space. Automatic event log summarization helps manage this by condensing older events into concise summaries while preserving recent, detailed events.

### Configuration

Enable summarization by setting `log_high_water_tokens` when creating an agent:

```python
agent = Agent(
    name="long_running_agent",
    log_high_water_tokens=20000,  # Trigger summarization at 20k tokens
    log_low_water_tokens=10000,   # Target 10k tokens after summarization (optional)
)
```

**Parameters:**
- **`log_high_water_tokens`**: When the event log exceeds this token count, automatic summarization triggers
- **`log_low_water_tokens`**: Target token count after summarization. Defaults to 50% of `log_high_water_tokens` if not specified

### How It Works: 3-Tier Context Management

agex uses a **3-tier rendering strategy** to maximize context efficiency:

1. **Full Detail** (events newer than threshold): Complete rendering with full images, deep nesting, verbose output
2. **Low Detail** (events older than threshold): Compressed rendering with image placeholders, shallow nesting
3. **Summarized** (oldest events): LLM-generated text summary replacing multiple old events

**Note on ratios:** When summarization triggers, the threshold is set to keep the newest ~25% at full detail. However, as new events accumulate between summarizations, they're all newer than the fixed threshold, so the ratio gradually shifts toward more full-detail events until the next summarization cycle.

#### Automatic Management Flow

1. **Monitoring**: Before each LLM call, the agent checks the total token count of all events
2. **Triggering**: If tokens exceed `log_high_water_tokens`, summarization runs
3. **Low-Detail Threshold**: Sets a **fixed timestamp threshold** (at 25th percentile by age) in the `SummaryEvent`
   - Events with `timestamp < threshold` render at low detail
   - Events with `timestamp >= threshold` render at full detail
   - As new events arrive, they're all newer than this fixed threshold (full detail) until next summarization
4. **Token Calculation**: Uses correct token counts when deciding what to keep:
   - Events newer than threshold: counted at `full_detail_tokens`
   - Events older than threshold: counted at `low_detail_tokens` (typically 25-50% of full)
5. **Summarization**: The LLM condenses the oldest events into a single `SummaryEvent` (see [Events](events.md#summaryevent))
6. **Replacement**: Old events are replaced with the summary, reducing total tokens below `log_low_water_tokens`

#### Low-Detail Rendering

For events older than the threshold, agex automatically applies budget-constrained rendering:

- **Images**: Replaced with `[Image]` text placeholders (saves ~1000 tokens per image)
- **Nested structures**: Reduced from depth 4 → depth 2 (e.g., dataframes, dicts)
- **List items**: Truncated more aggressively (25 items → 10 items)
- **Code and thinking**: Always full detail (already compact)

This means the agent can keep **significantly more event history** in context before needing to summarize.

### Why 50% Default?

The default `log_low_water_tokens` (50% of high water) is aggressive but intentional:

- **Cache invalidation cost**: Summarization breaks provider-side context caching, making it expensive
- **Maximize runway**: By reducing to 50%, you get maximum "runway" before the next summarization is needed
- **Rare but effective**: Summarization happens infrequently, but when it does, it significantly reduces token usage

### Example: Long-Running Analysis

```python
from agex import Agent, Versioned

# Agent configured for long-running tasks
analyst = Agent(
    name="data_analyst",
    primer="You are a data analyst working on complex, multi-step analyses.",
    log_high_water_tokens=15000,  # Summarize when log exceeds 15k tokens
    max_iterations=50,             # Allow many iterations
)

# Long-running task with persistent state
state = Versioned()

@analyst.task
def analyze_dataset(data_path: str) -> dict:  # type: ignore[return-value]
    """Perform comprehensive data analysis with many iterations."""
    pass

# As the agent works through many iterations, old events are automatically
# summarized to keep the context manageable
result = analyze_dataset("large_dataset.csv", state=state)
```

### Disabling Summarization

Summarization is **opt-in** by default. Simply don't set `log_high_water_tokens`:

```python
# No automatic summarization
agent = Agent(name="short_task_agent")
```

### Related

- **[Events](events.md#summaryevent)**: See `SummaryEvent` documentation
- **[State](state.md#garbage-collection-with-gcversioned)**: Similar garbage collection for persistent state

## Agent Registry
agex automatically registers all agents in a global registry to enable inter-agent communication. For **testing**, use `clear_agent_registry()` to prevent cross-contamination between test cases.

```python
from agex import clear_agent_registry
import pytest

# Typical pytest usage
@pytest.fixture(autouse=True)
def clear_agents():
    clear_agent_registry()
    yield
    clear_agent_registry()
```

## Next Steps

- **Registration Methods**: See [Registration](registration.md) for `.fn()`, `.cls()`, and `.module()` methods
- **Task Definition**: See [Task](task.md) for `@agent.task` usage
- **State Management**: See [State](state.md) for `Versioned` objects and persistent agent memory
- **Debugging**: See [View](view.md) for inspecting agent capabilities and execution state
