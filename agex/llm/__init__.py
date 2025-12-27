from typing import Any, Literal

from .config import get_llm_config
from .core import LLM, LLMResponse, TokenChunk
from .dummy_client import Dummy

# Optional imports for LLM providers
try:
    from .openai_client import OpenAI
except ImportError:
    OpenAI = None

try:
    from .anthropic_client import Anthropic
except ImportError:
    Anthropic = None

try:
    from .gemini_client import Gemini
except ImportError:
    Gemini = None

# Build __all__ dynamically based on available providers
__all__ = ["LLM", "Dummy", "connect_llm", "LLMResponse", "TokenChunk"]
if OpenAI is not None:
    __all__.append("OpenAI")
if Anthropic is not None:
    __all__.append("Anthropic")
if Gemini is not None:
    __all__.append("Gemini")


def connect_llm(
    provider: Literal["openai", "anthropic", "gemini", "dummy"] | None = None,
    model: str | None = None,
    timeout_seconds: float = 90.0,
    **kwargs: Any,
) -> LLM:
    """
    Factory function to get an LLM client.

    Resolves configuration from function parameters, global settings, and
    environment variables.

    Args:
        provider: LLM provider ("openai", "anthropic", "gemini", "dummy")
        model: Model name (e.g., "gpt-4.1-nano")
        timeout_seconds: API call timeout in seconds (default 90.0)
        **kwargs: Additional provider-specific arguments
    """
    # Resolve the full configuration from all sources
    config = get_llm_config(provider=provider, model=model, **kwargs)
    final_provider = config.get("provider")

    # Add timeout_seconds to config
    config["timeout_seconds"] = timeout_seconds

    # Dummy has special serialization for responses
    if final_provider == "dummy":
        # If responses are in serialized format (dicts from dump_config), use from_config
        if "responses" in config and config["responses"]:
            first = config["responses"][0]
            if isinstance(first, dict):
                # Serialized format - use from_config for proper reconstruction
                return Dummy.from_config(config)
        # Otherwise, pass through (for direct instantiation with LLMResponse objects)
        dummy_kwargs = {**config, **kwargs}
        dummy_kwargs.pop("provider", None)
        dummy_kwargs.pop("model", None)
        return Dummy(**dummy_kwargs)

    if final_provider == "anthropic":
        if Anthropic is None:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                'Install it with: pip install "agex[anthropic]"'
            )
        return Anthropic(**config)

    if final_provider == "gemini":
        if Gemini is None:
            raise ImportError(
                "Gemini provider requires the 'google-genai' package. "
                'Install it with: pip install "agex[gemini]"'
            )
        return Gemini(**config)

    if final_provider == "openai":
        if OpenAI is None:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                'Install it with: pip install "agex[openai]"'
            )
        return OpenAI(**config)

    # Build list of available providers for the error message
    available_providers = ["dummy"]
    if OpenAI is not None:
        available_providers.append("openai")
    if Anthropic is not None:
        available_providers.append("anthropic")
    if Gemini is not None:
        available_providers.append("gemini")

    raise ValueError(
        f"Unsupported provider: {final_provider}. Available providers are: {', '.join(available_providers)}"
    )
