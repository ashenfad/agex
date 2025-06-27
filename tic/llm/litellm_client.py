from dataclasses import asdict
from typing import List

import litellm

from ..tokenizers import get_tokenizer
from .core import LLMClient, Message


class LiteLLMClient(LLMClient):
    """LiteLLM-backed client for multi-provider LLM support."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4", **kwargs):
        """
        Initialize the LiteLLM client.

        Args:
            provider: The LLM provider (e.g., "openai", "anthropic", "azure")
            model: The model name (e.g., "gpt-4", "claude-3-sonnet")
            **kwargs: Additional arguments passed to LiteLLM (api_key, temperature, etc.)
        """
        self.provider = provider
        self._model = model
        self.kwargs = kwargs

        # Build model key for LiteLLM - some providers need prefixes
        if provider == "anthropic":
            self.model_key = (
                f"claude/{model}" if not model.startswith("claude/") else model
            )
        elif provider == "azure":
            # Azure typically needs azure/ prefix and deployment name
            self.model_key = (
                f"azure/{model}" if not model.startswith("azure/") else model
            )
        else:
            # OpenAI and most others can use model name directly
            self.model_key = model

        # Get compatible tokenizer for token estimation
        # Note: This might not be perfect for all providers, but good enough for estimation
        try:
            if model.startswith("gpt-"):
                self.tokenizer = get_tokenizer(model)
            else:
                # Fallback to a GPT tokenizer for non-OpenAI models
                # This is approximate but better than nothing
                self.tokenizer = get_tokenizer("gpt-4")
        except ValueError:
            # If no tokenizer available, fallback to GPT-4
            self.tokenizer = get_tokenizer("gpt-4")

        # Model registry for context windows
        # TODO: Could be moved to external config file
        self._context_windows = {
            "gpt-4": 128_000,
            "gpt-4-turbo": 128_000,
            "gpt-4-turbo-preview": 128_000,
            "gpt-3.5-turbo": 16_384,
            "claude-3-opus": 200_000,
            "claude-3-sonnet": 200_000,
            "claude-3-haiku": 200_000,
            "claude-3-5-sonnet": 200_000,
            "gemini-pro": 128_000,
            "gemini-1.5-pro": 1_000_000,
        }

    def complete(self, messages: List[Message], **kwargs) -> str:
        """
        Send messages to LLM via LiteLLM and return response content.

        Args:
            messages: List of Message objects with role and content
            **kwargs: Additional completion arguments (temperature, max_tokens, etc.)

        Returns:
            The LLM's response content as a string

        Raises:
            RuntimeError: If the completion request fails
        """
        try:
            # Convert Message objects to dicts for LiteLLM
            message_dicts = [asdict(msg) for msg in messages]

            # Merge instance kwargs with call-specific kwargs
            completion_kwargs = {**self.kwargs, **kwargs}

            response = litellm.completion(
                model=self.model_key, messages=message_dicts, **completion_kwargs
            )

            return response.choices[0].message.content  # type: ignore

        except Exception as e:
            # Wrap all LiteLLM exceptions in a consistent interface
            raise RuntimeError(f"LLM completion failed: {e}") from e

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens using the configured tokenizer.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(self.tokenizer.encode(text))

    @property
    def context_window(self) -> int:
        """
        Get context window for current model.

        Returns:
            Maximum context window size in tokens
        """
        # Try exact model match first
        if self._model in self._context_windows:
            return self._context_windows[self._model]

        # Try partial matches for model families
        for known_model, window_size in self._context_windows.items():
            if self._model.startswith(known_model):
                return window_size

        # Conservative default if model not recognized
        return 4096

    @property
    def model(self) -> str:
        """
        Get the model name being used.

        Returns:
            Model identifier string
        """
        return self._model
