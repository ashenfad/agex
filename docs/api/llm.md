# LLM Configuration

The `connect_llm()` factory function creates LLM clients for agent use. You can configure providers, models, and behavior through explicit parameters or environment variables.

## `connect_llm()` API

```python
from agex import connect_llm

llm = connect_llm(
    provider: Literal["openai", "anthropic", "gemini", "dummy"] | None = None,
    model: str | None = None,
    timeout_seconds: float = 90.0,
    **kwargs,  # Provider-specific arguments
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| None` | `None` | LLM provider: `"openai"`, `"anthropic"`, `"gemini"`, or `"dummy"`. If `None`, resolved from `AGEX_LLM_PROVIDER` env var. |
| `model` | `str \| None` | `None` | Model name (e.g., `"gpt-4.1-nano"`). If `None`, resolved from `AGEX_LLM_MODEL` env var or provider defaults. |
| `timeout_seconds` | `float` | `90.0` | API call timeout in seconds. |
| `**kwargs` | | | Additional arguments forwarded to the client (e.g., `api_key`, `temperature`, `max_tokens`). |

## Usage Patterns

### 1. Direct Instantiation (Recommended)

The clearest method is to create an LLM instance and pass it to the Agent's constructor:

```python
from agex import connect_llm, Agent

# Create a client for a specific provider
llm = connect_llm(provider="openai", model="gpt-4.1-nano")
agent = Agent(llm=llm)
```

### 2. Environment Variables (Default)

If you don't pass an `llm` to the Agent, agex creates one using environment variables:

```bash
export AGEX_LLM_PROVIDER="openai"
export AGEX_LLM_MODEL="gpt-4.1-nano"
export OPENAI_API_KEY="your-key-here"
```

```python
# Uses environment configuration
agent = Agent(primer="You are helpful.")
```

### 3. OpenAI-Compatible Endpoints (Ollama, etc.)

Connect to any OpenAI-compatible API:

```python
llm = connect_llm(
    provider="openai",
    model="qwen3-coder:30b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Placeholder for local services
)
```

> **Note:** We recommend `qwen3-coder:30b` for Ollama. Performance varies with other models.

## Provider-Specific Configuration

### Client vs Completion Arguments

`connect_llm` intelligently separates:
- **Client arguments**: `api_key`, `base_url`, `timeout`
- **Completion arguments**: `temperature`, `top_p`, `max_tokens`

```python
llm = connect_llm(
    provider="openai",
    model="gpt-4.1-nano",
    # Client args
    api_key="sk-...",
    timeout=30.0,
    # Completion args
    temperature=0.7,
    top_p=0.9,
)
```

### Gemini-Specific Features

```python
llm = connect_llm(
    provider="gemini",
    model="gemini-2.5-flash-preview-05-20",
    google_search=True,  # Enable Google Search grounding
    url_context=True,  # Enable URL context
)
```

## Testing with Dummy Client

For testing, use the `Dummy` client with predefined responses:

```python
from agex.llm import Dummy, LLMResponse

responses = [
    LLMResponse(thinking="Analyzing...", code='task_success(42)')
]
test_llm = Dummy(responses=responses)
agent = Agent(llm=test_llm)
```

## Retry and Timeout Behavior

- **`timeout_seconds`**: Per-API-call timeout, wired directly to each SDK's client (default: 90s)
- **`llm_max_retries`**: Set on Agent, retries failed completions with exponential backoff (default: 2)

SDK-level retries are disabled so that agex controls all retry behavior. Only transient errors are retried — rate limits, timeouts, connection errors, and internal server errors. Non-retryable errors (authentication failures, bad requests, etc.) fail immediately without consuming retry attempts.

```python
# Fast timeout with more retries
llm = connect_llm(provider="openai", timeout_seconds=30.0)
agent = Agent(llm=llm, llm_max_retries=5)
```

See [Error Handling - LLMFail](errors.md#llmfail-framework) for details on retry exhaustion behavior.

## Next Steps

- **[Agent](agent.md)**: Configure agents with LLM clients
- **[State](state.md)**: Add persistent memory to agents
- **[Host](host.md)**: Run agents on remote servers
