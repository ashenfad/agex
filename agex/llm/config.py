import os
from typing import Any, Dict


def get_llm_config(**overrides) -> Dict[str, Any]:
    """
    Get LLM configuration with a smart fallback chain.

    Priority (highest to lowest):
    1. Function overrides (e.g., in connect_llm)
    2. Environment variables (AGEX_LLM_*)
    3. Hard-coded defaults

    Args:
        **overrides: Direct configuration overrides.

    Returns:
        A complete LLM configuration dictionary.
    """
    # Start with hard-coded defaults
    config = {
        "provider": "dummy",
        "model": None,
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

    # Apply overrides (highest priority)
    # Filter out None values to allow selective overrides
    filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
    config.update(filtered_overrides)

    return config
