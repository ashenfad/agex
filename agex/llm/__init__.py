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
__all__ = ["DummyLLMClient"]
if OpenAIClient is not None:
    __all__.append("OpenAIClient")
if AnthropicClient is not None:
    __all__.append("AnthropicClient")
if GeminiClient is not None:
    __all__.append("GeminiClient")


def get_llm_client(
    provider: Literal["openai", "anthropic", "gemini", "dummy"] | None = None, **kwargs
) -> LLMClient:
    """
    Factory function to get an LLM client.
    """
    # If no provider specified, get it from configuration
    if provider is None:
        config = get_llm_config(**kwargs)
        provider = config["provider"]

    if provider == "dummy":
        return DummyLLMClient(**kwargs)

    if provider == "anthropic":
        if AnthropicClient is None:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install 'agex[anthropic]'"
            )
        config = get_llm_config(**kwargs)
        return AnthropicClient(**config)

    if provider == "gemini":
        if GeminiClient is None:
            raise ImportError(
                "Gemini provider requires the 'google-generativeai' package. "
                "Install it with: pip install 'agex[gemini]'"
            )
        config = get_llm_config(**kwargs)
        return GeminiClient(**config)

    if provider == "openai":
        if OpenAIClient is None:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install 'agex[openai]'"
            )
        config = get_llm_config(**kwargs)
        return OpenAIClient(**config)

    # Build list of available providers
    available_providers = ["dummy"]
    if OpenAIClient is not None:
        available_providers.append("openai")
    if AnthropicClient is not None:
        available_providers.append("anthropic")
    if GeminiClient is not None:
        available_providers.append("gemini")

    raise ValueError(
        f"Unsupported provider: {provider}. Available providers are: {', '.join(available_providers)}"
    )
