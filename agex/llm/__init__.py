from typing import Literal

from .config import get_llm_config
from .core import LLMClient
from .dummy_client import DummyLLMClient
from .openai_client import OpenAIClient

__all__ = ["DummyLLMClient", "OpenAIClient"]


def get_llm_client(
    provider: Literal["openai", "dummy"] = "openai", **kwargs
) -> LLMClient:
    """
    Factory function to get an LLM client.
    """
    if provider == "dummy":
        return DummyLLMClient(**kwargs)

    if provider != "openai":
        raise ValueError("Only 'openai' and 'dummy' providers are currently supported.")

    config = get_llm_config(**kwargs)

    return OpenAIClient(**config)
