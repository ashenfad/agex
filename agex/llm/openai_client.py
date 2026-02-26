from typing import Any, AsyncIterator, Iterator, List

import openai

from agex.agent.events import Event
from agex.llm.core import (
    LLM,
    TokenChunk,
)
from agex.llm.xml import XML_FORMAT_PRIMER, tokenize_xml_stream
from agex.tokenizers import get_tokenizer

# Define keys for client setup vs. completion
CLIENT_CONFIG_KEYS = {"api_key", "base_url", "organization", "timeout", "max_retries"}


def _format_message_for_openai(message: dict[str, Any]) -> dict:
    """
    Convert generic message dict to OpenAI's format.

    Handles multimodal content (images) conversion.

    Note: All images are converted to PNG format by the rendering layer
    (serialize_image_to_base64) before reaching this function.
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


class OpenAI(LLM):
    """OpenAI LLM provider with native structured outputs."""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        timeout_seconds: float = 90.0,
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

        # Wire timeout and disable SDK retries (agex handles retries)
        client_kwargs.setdefault("timeout", timeout_seconds)
        client_kwargs.setdefault("max_retries", 0)

        self._model = model
        self._kwargs = completion_kwargs
        self._timeout_seconds = timeout_seconds
        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = openai.AsyncOpenAI(**client_kwargs)
        self.tokenizer = get_tokenizer(model)

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds for each API call."""
        return self._timeout_seconds

    def dump_config(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            **self._kwargs,  # Include other completion args
        }

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """
        Stream tokens from OpenAI using XML format.

        Uses standard streaming API with XML parsing for token-level updates.
        """
        from agex.render.xml import render_events_as_xml

        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Use XML rendering for streaming (instead of structured outputs)
        messages_dicts = render_events_as_xml(events)

        # Add system message with XML format instructions
        system_with_format = f"{system}\n\n{XML_FORMAT_PRIMER}"
        full_messages = [
            {"role": "system", "content": system_with_format}
        ] + messages_dicts

        # Request usage data on the final chunk
        stream = self.client.chat.completions.create(
            model=self._model,
            messages=[_format_message_for_openai(msg) for msg in full_messages],  # type: ignore
            stream=True,
            stream_options={"include_usage": True},
            **request_kwargs,
        )

        # Generator for raw text chunks; capture usage from final chunk
        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        def raw_chunks() -> Iterator[str]:
            for chunk in stream:
                if chunk.usage is not None:
                    usage_holder["input_tokens"] = chunk.usage.prompt_tokens
                    usage_holder["output_tokens"] = chunk.usage.completion_tokens
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

        # Parse XML stream into TokenChunks
        yield from tokenize_xml_stream(raw_chunks())

        # Yield final usage token
        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

    async def acomplete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """Async version of complete_stream."""
        from agex.llm.xml import atokenize_xml_stream
        from agex.render.xml import render_events_as_xml

        request_kwargs = {**self._kwargs, **kwargs}
        messages_dicts = render_events_as_xml(events)

        system_with_format = f"{system}\n\n{XML_FORMAT_PRIMER}"
        full_messages = [
            {"role": "system", "content": system_with_format}
        ] + messages_dicts

        stream = await self.async_client.chat.completions.create(
            model=self._model,
            messages=[_format_message_for_openai(msg) for msg in full_messages],  # type: ignore
            stream=True,
            stream_options={"include_usage": True},
            **request_kwargs,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        async def raw_chunks():
            async for chunk in stream:
                if chunk.usage is not None:
                    usage_holder["input_tokens"] = chunk.usage.prompt_tokens
                    usage_holder["output_tokens"] = chunk.usage.completion_tokens
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

        async for token in atokenize_xml_stream(raw_chunks()):
            yield token

        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

    def summarize(self, system: str, content: str | List[Event], **kwargs) -> str:
        """Send a summarization request to OpenAI (text or events with multimodal)."""
        request_kwargs = {**self._kwargs, **kwargs}

        is_multimodal, processed = self._prepare_summarization_content(content)

        if is_multimodal:
            full_messages = [{"role": "system", "content": system}] + processed
        else:
            full_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": processed},
            ]

        response = self.client.chat.completions.create(
            model=self._model,
            messages=[_format_message_for_openai(msg) for msg in full_messages],  # type: ignore
            **request_kwargs,
        )
        result = response.choices[0].message.content
        if isinstance(result, list):
            texts = []
            for part in result:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            return "".join(texts)
        return result or ""

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "OpenAI"
