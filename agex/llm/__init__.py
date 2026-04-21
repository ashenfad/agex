from typing import Any, Literal

from .config import get_llm_config
from .core import LLM, LLMResponse, TokenChunk
from .dummy_client import Dummy

__all__ = [
    "LLM",
    "Dummy",
    "connect_llm",
    "LLMResponse",
    "TokenChunk",
    # Provider classes are resolved lazily via ``__getattr__`` so that
    # importing heavy SDKs (google-genai, anthropic, openai, …) is
    # deferred until a caller actually touches the name. This keeps
    # ``import agex`` fast — a provider like google-genai at module load
    # transitively pulls PIL and more.
    "OpenAI",
    "Anthropic",
    "Gemini",
    "PyfetchOpenAI",
    "PyfetchAnthropic",
]


# Map public provider name → (module suffix, class name). Used by both
# ``__getattr__`` below and ``connect_llm`` so both code paths route
# through the same lazy-import seam.
_PROVIDERS: dict[str, tuple[str, str]] = {
    "OpenAI": ("openai_client", "OpenAI"),
    "Anthropic": ("anthropic_client", "Anthropic"),
    "Gemini": ("gemini_client", "Gemini"),
    "PyfetchOpenAI": ("pyfetch_openai", "PyfetchOpenAI"),
    "PyfetchAnthropic": ("pyfetch_anthropic", "PyfetchAnthropic"),
}

_INSTALL_HINT: dict[str, str] = {
    "anthropic": 'Install with: pip install "agex[anthropic]"',
    "gemini": 'Install with: pip install "agex[gemini]"',
    "openai": 'Install with: pip install "agex[openai]"',
}


def _load_provider(public_name: str):
    """Lazily import a provider class. Returns None when the underlying
    SDK isn't installed — callers use this to decide whether to raise
    a helpful install hint.

    Honors explicit module-level overrides: tests that set
    ``agex.llm.OpenAI = None`` to simulate a missing SDK still work,
    because we check ``globals()`` first before attempting the import.
    """
    # Explicit override wins (e.g. ``patch("agex.llm.OpenAI", None)``).
    mod_globals = globals()
    if public_name in mod_globals:
        return mod_globals[public_name]

    entry = _PROVIDERS.get(public_name)
    if entry is None:
        return None
    module_suffix, class_name = entry
    try:
        module = __import__(
            f"agex.llm.{module_suffix}",
            fromlist=[class_name],
        )
    except ImportError:
        return None
    return getattr(module, class_name, None)


def __getattr__(name: str):
    """Lazy attribute access: ``from agex.llm import OpenAI`` works as
    before, but the corresponding SDK isn't imported until the name is
    actually requested.
    """
    if name in _PROVIDERS:
        cls = _load_provider(name)
        if cls is None:
            # Preserve the old ``None sentinel`` behavior so callers
            # that explicitly check ``if OpenAI is None`` still work.
            return None
        return cls
    raise AttributeError(f"module 'agex.llm' has no attribute {name!r}")


def connect_llm(
    provider: Literal[
        "openai",
        "anthropic",
        "gemini",
        "pyfetch_openai",
        "pyfetch_anthropic",
        "dummy",
    ]
    | None = None,
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

    # Providers map config-name → public class name. Keep this in one
    # place so both ``__getattr__`` and ``connect_llm`` route through
    # the same lazy-load seam.
    CONFIG_TO_CLASS = {
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "openai": "OpenAI",
        "pyfetch_openai": "PyfetchOpenAI",
        "pyfetch_anthropic": "PyfetchAnthropic",
    }

    class_name = CONFIG_TO_CLASS.get(final_provider or "")
    if class_name is not None:
        cls = _load_provider(class_name)
        if cls is None:
            hint = _INSTALL_HINT.get(final_provider or "", "")
            msg = f"{class_name} provider could not be loaded." + (
                f" {hint}" if hint else ""
            )
            raise ImportError(msg)
        return cls(**config)

    # Build list of available providers for the error message. Probe
    # each lazily — we don't want this to import the SDKs either.
    available = ["dummy"]
    for cfg_name, cls_name in CONFIG_TO_CLASS.items():
        if _load_provider(cls_name) is not None:
            available.append(cfg_name)

    raise ValueError(
        f"Unsupported provider: {final_provider}. "
        f"Available providers are: {', '.join(available)}"
    )
