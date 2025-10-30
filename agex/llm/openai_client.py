from typing import Any, List

import openai

from agex.agent.events import Event
from agex.llm.core import LLMClient, LLMResponse
from agex.tokenizers import get_tokenizer

# Define keys for client setup vs. completion
CLIENT_CONFIG_KEYS = {"api_key", "base_url", "organization", "timeout"}


def _format_message_for_openai(message: dict[str, Any]) -> dict:
    """
    Convert generic message dict to OpenAI's format.

    Handles multimodal content (images) conversion.

    Note: All images are converted to PNG format by the rendering layer
    (StreamRenderer._serialize_image_to_base64) before reaching this function.
    """
    if isinstance(message.get("content"), list):
        # Multimodal message
        content_parts = []
        for part in message["content"]:
            if part["type"] == "text":
                content_parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{part['image_data']}"
                        },
                    }
                )
        return {"role": message["role"], "content": content_parts}
    else:
        # Text message
        return message


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

    def complete(self, system: str, events: List[Event], **kwargs) -> LLMResponse:
        """
        Send events to OpenAI and return a structured response using native structured outputs.
        """
        from agex.render.events import render_events_as_markdown

        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Use rendering helper to convert events to markdown messages
        max_tokens = request_kwargs.get("max_tokens", 4096)
        messages_dicts = render_events_as_markdown(events, self._model, max_tokens)

        # Add system message at the beginning
        full_messages = [{"role": "system", "content": system}] + messages_dicts

        try:
            # Use OpenAI's native structured outputs with beta.chat.completions.parse
            response = self.client.beta.chat.completions.parse(
                model=self._model,
                messages=[_format_message_for_openai(msg) for msg in full_messages],  # type: ignore
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

    def summarize(self, system: str, content: str, **kwargs) -> str:
        """Send a simple text summarization request to OpenAI."""
        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                **request_kwargs,
            )
            result = response.choices[0].message.content
            if isinstance(result, list):
                # When OpenAI returns content parts, join text parts
                texts = []
                for part in result:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                return "".join(texts)
            return result or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI text completion failed: {e}") from e

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "OpenAI"
