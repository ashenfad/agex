from typing import Any, Iterator, List

import anthropic
from anthropic.types import TextBlockParam

from agex.agent.events import Event
from agex.llm.core import LLMClient, LLMResponse, TokenChunk
from agex.llm.xml import XML_FORMAT_PRIMER, tokenize_xml_stream

# Define keys for client setup vs. completion
CLIENT_CONFIG_KEYS = {"api_key", "timeout"}
MAX_TOKENS = 2**14
CACHE_TTL = "1h"


def _with_cache(messages: list[dict]) -> list[dict]:
    messages[-1]["cache_control"] = {"type": "ephemeral", "ttl": CACHE_TTL}
    return messages


def _format_message_for_anthropic(
    is_last_message: bool, message: dict[str, Any]
) -> dict:
    """
    Convert generic message dict to Anthropic's format.

    Handles multimodal content (images) conversion.

    Note: All images are converted to PNG format by the rendering layer
    (StreamRenderer._serialize_image_to_base64) before reaching this function.
    """
    content_parts: list[dict] = []
    if isinstance(message.get("content"), list):
        # Multimodal message
        for part in message["content"]:
            if part["type"] == "text":
                content_parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                content_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": part["image_data"],
                        },
                    }
                )
    else:
        content_parts.append({"type": "text", "text": message["content"]})

    if is_last_message:
        _with_cache(content_parts)
    return {"role": message["role"], "content": content_parts}


class AnthropicClient(LLMClient):
    """Client for Anthropic's API using tool calling for structured outputs."""

    def __init__(self, model: str = "claude-3-sonnet-20240229", **kwargs):
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
        self.client = anthropic.Anthropic(**client_kwargs)

    def complete(self, system: str, events: List[Event], **kwargs) -> LLMResponse:
        """
        Send events to Anthropic and return a structured response using tool calling.
        """
        from agex.render.events import render_events_as_markdown

        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Use rendering helper to convert events to markdown messages
        messages_dicts = render_events_as_markdown(events)

        # Convert to Anthropic format
        conversation_messages = [
            _format_message_for_anthropic(index == len(messages_dicts) - 1, msg)
            for index, msg in enumerate(messages_dicts)
        ]

        # Define the structured response tool
        structured_response_tool = {
            "name": "structured_response",
            "description": "Respond with thinking and code in a structured format",
            "input_schema": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Your natural language thinking about the task",
                    },
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute",
                    },
                },
                "required": ["thinking", "code"],
            },
        }

        try:
            # Set default max_tokens if not provided
            if "max_tokens" not in request_kwargs:
                request_kwargs["max_tokens"] = MAX_TOKENS

            # Make the API call with tool calling
            # Create API call with system message
            response = self.client.messages.create(
                model=self._model,
                system=system,
                messages=conversation_messages,
                tools=[structured_response_tool],
                tool_choice={"type": "tool", "name": "structured_response"},
                **request_kwargs,
            )

            # Extract the structured response from tool use
            if not response.content or len(response.content) == 0:
                raise RuntimeError("Anthropic returned empty response")

            # Look for tool use in the response
            tool_use = None
            for content_block in response.content:
                if (
                    content_block.type == "tool_use"
                    and content_block.name == "structured_response"
                ):
                    tool_use = content_block
                    break

            if tool_use is None:
                raise RuntimeError("Anthropic did not return expected tool use")

            # Extract thinking and code from tool input
            tool_input = tool_use.input
            thinking = tool_input.get("thinking", "")
            code = tool_input.get("code", "")

            return LLMResponse(thinking=thinking, code=code)

        except Exception as e:
            raise RuntimeError(f"Anthropic completion failed: {e}") from e

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """
        Stream tokens from Anthropic using XML format.

        Uses standard streaming API with XML parsing for token-level updates.
        """
        from agex.render.xml import render_events_as_xml

        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Use XML rendering for streaming (instead of tool calling)
        messages_dicts = render_events_as_xml(events)

        # Convert to Anthropic format
        conversation_messages = [
            _format_message_for_anthropic(index == len(messages_dicts) - 1, msg)
            for index, msg in enumerate(messages_dicts)
        ]

        # Add system message with XML format instructions
        system_with_format = f"{system}\n\n{XML_FORMAT_PRIMER}"
        system_block = TextBlockParam(
            type="text",
            text=system_with_format,
            cache_control={"type": "ephemeral", "ttl": CACHE_TTL},
        )

        try:
            # Set default max_tokens if not provided
            if "max_tokens" not in request_kwargs:
                request_kwargs["max_tokens"] = MAX_TOKENS

            # Use standard streaming API (not tool calling)
            stream = self.client.messages.stream(
                model=self._model,
                system=[system_block],
                messages=conversation_messages,
                **request_kwargs,
            )

            # Generator for raw text chunks from Anthropic
            def raw_chunks() -> Iterator[str]:
                with stream as message_stream:
                    for text in message_stream.text_stream:
                        yield text

            # Parse XML stream into TokenChunks
            yield from tokenize_xml_stream(raw_chunks())

        except Exception as e:
            raise RuntimeError(f"Anthropic streaming completion failed: {e}") from e

    def summarize(self, system: str, content: str | List[Event], **kwargs) -> str:
        """Send a summarization request to Anthropic (text or events with multimodal)."""
        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Prepare content (text or events)
        is_multimodal, processed = self._prepare_summarization_content(content)

        if is_multimodal:
            # processed is messages list from events
            # Convert to Anthropic format
            conversation_messages = [
                _format_message_for_anthropic(index == len(processed) - 1, msg)
                for index, msg in enumerate(processed)
            ]
        else:
            # processed is plain text
            conversation_messages = [
                {"role": "user", "content": [{"type": "text", "text": processed}]}
            ]

        try:
            if "max_tokens" not in request_kwargs:
                request_kwargs["max_tokens"] = MAX_TOKENS

            response = self.client.messages.create(
                model=self._model,
                system=system,
                messages=conversation_messages,
                **request_kwargs,
            )
            # Concatenate text parts from content blocks
            texts: list[str] = []
            for block in response.content or []:
                if getattr(block, "type", None) == "text":
                    texts.append(getattr(block, "text", ""))
            return "".join(texts)
        except Exception as e:
            raise RuntimeError(f"Anthropic summarization failed: {e}") from e

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Anthropic"
