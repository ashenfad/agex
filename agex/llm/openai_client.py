from dataclasses import asdict
from typing import List

import instructor
import openai

from agex.llm.core import LLMClient, LLMResponse, Message
from agex.tokenizers import get_tokenizer


class OpenAIClient(LLMClient):
    """Client for OpenAI's API."""

    def __init__(self, model: str = "gpt-4.1-nano", **kwargs):
        self._model = model
        self._kwargs = kwargs
        # Patch the OpenAI client with instructor
        self.client = instructor.from_openai(openai.OpenAI())
        self.tokenizer = get_tokenizer(model)
        self._context_windows = {
            "gpt-4.1-nano": 8192,
        }

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Send messages to OpenAI and return a structured response.
        """
        # Combine kwargs, giving precedence to method-level ones
        # request_kwargs = {**self._kwargs, **kwargs}
        print("ADAM ---- messages....")
        for msg in messages[-3:]:
            print(msg)
            print("--------------")
        try:
            # Pydantic models can be passed directly to the patched client
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[asdict(msg) for msg in messages],  # type: ignore
                response_model=LLMResponse,
            )
            print("ADAM ---- response....")
            print(response.thinking)
            print(response.code)
            print("--------------")
            return response
        except Exception as e:
            raise RuntimeError(f"OpenAI completion failed: {e}") from e

    def estimate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @property
    def context_window(self) -> int:
        return self._context_windows.get(self.model, 8192)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "OpenAI"
