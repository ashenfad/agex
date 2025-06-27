import os
from typing import Any, Dict

# Global LLM configuration storage
_GLOBAL_LLM_CONFIG: Dict[str, Any] = {}


def get_llm_config(**overrides) -> Dict[str, Any]:
    """
    Get LLM configuration with smart fallback chain.

    Priority (highest to lowest):
    1. Function overrides (agent constructor args)
    2. Global configuration (tic.configure_llm)
    3. Environment variables (AGEX_LLM_*)
    4. Hard-coded defaults

    Args:
        **overrides: Direct configuration overrides

    Returns:
        Complete LLM configuration dictionary
    """
    # Start with hard-coded defaults
    config = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
    }

    # Apply environment variables
    env_mapping = {
        "AGEX_LLM_PROVIDER": "provider",
        "AGEX_LLM_MODEL": "model",
        "AGEX_LLM_TEMPERATURE": "temperature",
        "AGEX_LLM_MAX_TOKENS": "max_tokens",
        "AGEX_LLM_TOP_P": "top_p",
    }

    for env_var, config_key in env_mapping.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Handle type conversion for numeric values
            if config_key in ("temperature", "top_p"):
                config[config_key] = float(env_value)
            elif config_key == "max_tokens":
                config[config_key] = int(env_value)
            else:
                config[config_key] = env_value

    # Apply global configuration
    config.update(_GLOBAL_LLM_CONFIG)

    # Apply overrides (highest priority)
    # Filter out None values to allow selective overrides
    filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
    config.update(filtered_overrides)

    return config


def configure_llm(**kwargs):
    """
    Set global LLM defaults for all new agents.

    Example:
        tic.configure_llm(provider="anthropic", model="claude-3-sonnet")

    Args:
        **kwargs: LLM configuration parameters
    """
    global _GLOBAL_LLM_CONFIG
    _GLOBAL_LLM_CONFIG.update(kwargs)


def get_global_llm_config() -> Dict[str, Any]:
    """
    Get the current global LLM configuration.

    Returns:
        Copy of the global configuration dictionary
    """
    return _GLOBAL_LLM_CONFIG.copy()


def reset_llm_config():
    """
    Reset global LLM configuration to empty.
    Useful for testing.
    """
    global _GLOBAL_LLM_CONFIG
    _GLOBAL_LLM_CONFIG = {}
