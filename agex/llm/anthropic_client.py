from typing import List

import anthropic

from agex.llm.core import (
    ContentPart,
    LLMClient,
    LLMResponse,
    Message,
    MultimodalMessage,
    TextMessage,
    TextPart,
)


def _format_content(message: Message) -> List[dict]:
    """Format a Message object's content into the list structure Anthropic expects."""
    content_parts: List[ContentPart] = []
    if isinstance(message, TextMessage):
        content_parts = [TextPart(text=message.content)]
    elif isinstance(message, MultimodalMessage):
        content_parts = message.content

    formatted_parts = []
    for part in content_parts:
        if part.type == "text":
            formatted_parts.append({"type": "text", "text": part.text})
        elif part.type == "image":
            formatted_parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": part.image,
                    },
                }
            )
    return formatted_parts


class AnthropicClient(LLMClient):
    """Client for Anthropic's API using tool calling for structured outputs."""

    def __init__(self, model: str = "claude-3-sonnet-20240229", **kwargs):
        kwargs = kwargs.copy()
        kwargs.pop("provider", None)
        self._model = model
        self._kwargs = kwargs
        self.client = anthropic.Anthropic()
        self._context_windows = {
            "claude-3-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-5-haiku-20241022": 200000,
        }

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Send messages to Anthropic and return a structured response using tool calling.
        """
        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Separate system message from conversation messages
        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                # System message content must be a simple string for Anthropic
                if isinstance(msg, MultimodalMessage):
                    raise TypeError("Anthropic system messages cannot contain images.")
                if system_message is None:
                    system_message = msg.content
                else:
                    system_message += "\n\n" + msg.content
            else:
                conversation_messages.append(
                    {"role": msg.role, "content": _format_content(msg)}
                )

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
            print(
                "ADAM ------------------------ calling anthropic with model",
                self._model,
            )
            # Set default max_tokens if not provided
            if "max_tokens" not in request_kwargs:
                request_kwargs["max_tokens"] = 4096

            # Make the API call with tool calling
            # Only include system parameter if we have a system message
            api_kwargs = {
                "model": self._model,
                "messages": conversation_messages,
                "tools": [structured_response_tool],
                "tool_choice": {"type": "tool", "name": "structured_response"},
                **request_kwargs,
            }
            if system_message is not None:
                api_kwargs["system"] = system_message

            response = self.client.messages.create(**api_kwargs)

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

    def estimate_tokens(self, text: str) -> int:
        # Anthropic's rough estimate: ~4 characters per token
        return len(text) // 4

    @property
    def context_window(self) -> int:
        return self._context_windows.get(self._model, 200000)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Anthropic"
