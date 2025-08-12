from dataclasses import asdict
from typing import List

import openai

from agex.llm.core import (
    LLMClient,
    LLMResponse,
    Message,
    MultimodalMessage,
    TextMessage,
)
from agex.tokenizers import get_tokenizer

# Define keys for client setup vs. completion
CLIENT_CONFIG_KEYS = {"api_key", "base_url", "organization", "timeout"}


def _format_message(message: Message) -> dict:
    """Format a Message object into the dictionary structure OpenAI expects."""
    if isinstance(message, TextMessage):
        return asdict(message)

    if isinstance(message, MultimodalMessage):
        content_parts = []
        for part in message.content:
            if part.type == "text":
                content_parts.append({"type": "text", "text": part.text})
            elif part.type == "image":
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{part.image}"},
                    }
                )
        return {"role": message.role, "content": content_parts}

    raise TypeError(f"Unsupported message type: {type(message)}")


class OpenAIClient(LLMClient):
    """Client for OpenAI's API with native structured outputs."""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        **kwargs,
    ):
        kwargs.pop("provider", None)
        client_kwargs = {}
        completion_kwargs = {}
        for key, value in kwargs.items():
            if key in CLIENT_CONFIG_KEYS:
                client_kwargs[key] = value
            else:
                completion_kwargs[key] = value

        self._model = model
        self._kwargs = completion_kwargs
        self.client = openai.OpenAI(**client_kwargs)
        self.tokenizer = get_tokenizer(model)

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Send messages to OpenAI and return a structured response using native structured outputs.
        """
        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        try:
            # Use OpenAI's native structured outputs with beta.chat.completions.parse
            response = self.client.beta.chat.completions.parse(
                model=self._model,
                messages=[_format_message(msg) for msg in messages],  # type: ignore
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

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "OpenAI"
