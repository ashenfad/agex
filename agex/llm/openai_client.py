from dataclasses import asdict
from typing import List

import openai

from agex.llm.core import LLMClient, LLMResponse, Message
from agex.tokenizers import get_tokenizer


class OpenAIClient(LLMClient):
    """Client for OpenAI's API with native structured outputs."""

    def __init__(self, model: str = "gpt-4.1-nano", **kwargs):
        kwargs = kwargs.copy()
        kwargs.pop("provider", None)
        self._model = model
        self._kwargs = kwargs
        # Use the standard OpenAI client - no instructor patching needed
        self.client = openai.OpenAI()
        self.tokenizer = get_tokenizer(model)
        self._context_windows = {
            "gpt-4.1": 128000,
            "gpt-4.1-nano": 128000,
        }

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Send messages to OpenAI and return a structured response using native structured outputs.
        """
        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        try:
            # Use OpenAI's native structured outputs with beta.chat.completions.parse
            response = self.client.chat.completions.parse(
                model=self._model,
                messages=[asdict(msg) for msg in messages],  # type: ignore
                response_format=LLMResponse,
                **request_kwargs,
            )

            # Extract the parsed response
            parsed_response = response.choices[0].message.parsed
            if parsed_response is None:
                raise RuntimeError("OpenAI returned None for parsed response")
            return parsed_response

        except Exception as e:
            raise RuntimeError(f"OpenAI completion failed: {e}") from e

    def estimate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @property
    def context_window(self) -> int:
        return self._context_windows.get(self.model, 128000)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "OpenAI"
