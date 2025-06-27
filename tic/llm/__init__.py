from .core import LLMClient
from .litellm_client import LiteLLMClient


def get_llm_client(
    provider: str = "openai", model: str = "gpt-4", **kwargs
) -> LLMClient:
    """
    Factory function to get the appropriate LLM client for a given provider and model.

    Args:
        provider: The LLM provider (e.g., "openai", "anthropic", "azure")
        model: The model name (e.g., "gpt-4", "claude-3-sonnet")
        **kwargs: Additional arguments passed to the client

    Returns:
        An LLMClient instance

    Raises:
        ValueError: If the provider/model combination is not supported
    """
    # For now, we only support LiteLLM for all providers
    # In the future, this can be extended for provider-specific clients
    return LiteLLMClient(provider=provider, model=model, **kwargs)
