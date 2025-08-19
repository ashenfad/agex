from typing import Literal

from .config import get_llm_config
from .core import LLMClient
from .dummy_client import DummyLLMClient

# Optional imports for LLM providers
try:
    from .openai_client import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from .anthropic_client import AnthropicClient
except ImportError:
    AnthropicClient = None

try:
    from .gemini_client import GeminiClient
except ImportError:
    GeminiClient = None

# Build __all__ dynamically based on available clients
__all__ = ["DummyLLMClient", "connect_llm"]
if OpenAIClient is not None:
    __all__.append("OpenAIClient")
if AnthropicClient is not None:
    __all__.append("AnthropicClient")
if GeminiClient is not None:
    __all__.append("GeminiClient")


def connect_llm(
    provider: Literal["openai", "anthropic", "gemini", "dummy"] | None = None,
    model: str | None = None,
    **kwargs,
) -> LLMClient:
    """
    Factory function to get an LLM client.

    Resolves configuration from function parameters, global settings, and
    environment variables.
    """
    # Resolve the full configuration from all sources
    config = get_llm_config(provider=provider, model=model, **kwargs)
    final_provider = config.get("provider")

    # The DummyLLMClient has a unique `responses` kwarg that other clients do not.
    # We pass the original kwargs to it to preserve this behavior.
    if final_provider == "dummy":
        return DummyLLMClient(**kwargs)

    if final_provider == "anthropic":
        if AnthropicClient is None:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                'Install it with: pip install "agex[anthropic]"'
            )
        return AnthropicClient(**config)

    if final_provider == "gemini":
        if GeminiClient is None:
            raise ImportError(
                "Gemini provider requires the 'google-generativeai' package. "
                'Install it with: pip install "agex[gemini]"'
            )
        return GeminiClient(**config)

    if final_provider == "openai":
        if OpenAIClient is None:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                'Install it with: pip install "agex[openai]"'
            )
        return OpenAIClient(**config)

    # Build list of available providers for the error message
    available_providers = ["dummy"]
    if OpenAIClient is not None:
        available_providers.append("openai")
    if AnthropicClient is not None:
        available_providers.append("anthropic")
    if GeminiClient is not None:
        available_providers.append("gemini")

    raise ValueError(
        f"Unsupported provider: {final_provider}. Available providers are: {', '.join(available_providers)}"
    )
