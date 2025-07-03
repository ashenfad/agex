import json
from typing import List

import google.generativeai as genai

from agex.llm.core import LLMClient, LLMResponse, Message


class GeminiClient(LLMClient):
    """Client for Google's Gemini API with structured outputs."""

    def __init__(self, model: str = "gemini-1.5-flash", **kwargs):
        kwargs = kwargs.copy()
        kwargs.pop("provider", None)
        self._model = model
        self._kwargs = kwargs

        # Initialize the Gemini client
        self.client = genai.GenerativeModel(model)

        self._context_windows = {
            "gemini-1.5-flash": 1048576,  # 1M tokens
            "gemini-1.5-pro": 2097152,  # 2M tokens
            "gemini-1.0-pro": 32768,  # 32K tokens
        }

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Send messages to Gemini and return a structured response.
        """
        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_to_gemini_format(messages)

        # Define the structured output schema
        response_schema = {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string",
                    "description": "Your natural language thinking about the task",
                },
                "code": {"type": "string", "description": "The Python code to execute"},
            },
            "required": ["thinking", "code"],
        }

        try:
            # Configure generation with structured output
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                **request_kwargs,
            )

            # Generate response
            response = self.client.generate_content(
                gemini_messages, generation_config=generation_config
            )

            # Parse the JSON response
            if not response.text:
                raise RuntimeError("Gemini returned empty response")

            try:
                parsed_response = json.loads(response.text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse Gemini JSON response: {e}")

            # Extract thinking and code
            thinking = parsed_response.get("thinking", "")
            code = parsed_response.get("code", "")

            return LLMResponse(thinking=thinking, code=code)

        except Exception as e:
            raise RuntimeError(f"Gemini completion failed: {e}") from e

    def _convert_messages_to_gemini_format(self, messages: List[Message]) -> List[dict]:
        """Convert agex Message objects to Gemini's expected format."""
        gemini_messages = []
        system_content = None

        for message in messages:
            if message.role == "system":
                # Gemini handles system messages differently - they're part of the model configuration
                # For now, we'll prepend system messages to the first user message
                if system_content is None:
                    system_content = message.content
                else:
                    system_content += "\n\n" + message.content
            elif message.role == "user":
                content = message.content
                if system_content:
                    # Prepend system message to first user message
                    content = f"System: {system_content}\n\nUser: {content}"
                    system_content = None  # Only add once
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
            elif message.role == "assistant":
                gemini_messages.append(
                    {
                        "role": "model",  # Gemini uses "model" instead of "assistant"
                        "parts": [{"text": message.content}],
                    }
                )

        return gemini_messages

    def estimate_tokens(self, text: str) -> int:
        # Gemini's rough estimate: ~4 characters per token (similar to other models)
        return len(text) // 4

    @property
    def context_window(self) -> int:
        return self._context_windows.get(self._model, 1048576)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Google Gemini"
