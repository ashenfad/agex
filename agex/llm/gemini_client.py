import json
from typing import Any, Iterator, List, cast

import google.generativeai as genai

from agex.agent.events import Event
from agex.llm.core import LLMClient, LLMResponse, TokenChunk
from agex.llm.xml import XML_FORMAT_PRIMER, tokenize_xml_stream

# Define keys for client setup vs. completion
CLIENT_CONFIG_KEYS = {"api_key"}


class GeminiClient(LLMClient):
    """Client for Google's Gemini API with structured outputs."""

    def __init__(self, model: str = "gemini-1.5-flash", **kwargs):
        kwargs = kwargs.copy()
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

        # Configure API key if provided (note: this affects global state)
        if "api_key" in client_kwargs:
            genai.configure(api_key=client_kwargs["api_key"])  # type: ignore[attr-defined]

        self.client = genai.GenerativeModel(model_name=model)  # type: ignore[attr-defined]

    def complete(self, system: str, events: List[Event], **kwargs) -> LLMResponse:
        """
        Send events to Gemini and return a structured response.
        """
        from agex.render.events import render_events_as_markdown

        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Use rendering helper to convert events to markdown messages
        messages_dicts = render_events_as_markdown(events)

        # Convert to Gemini format (with system prepended)
        gemini_messages = self._convert_messages_to_gemini_format(
            system, messages_dicts
        )

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
            generation_config = genai.GenerationConfig(  # type: ignore
                response_mime_type="application/json",
                response_schema=response_schema,
                **request_kwargs,
            )
            # Generate response
            # Gemini expects a chat-style list of dict parts; typing stubs may not align.
            response = self.client.generate_content(
                cast(Any, gemini_messages), generation_config=generation_config
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

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """
        Stream tokens from Gemini using XML format.

        Uses standard streaming API with XML parsing for token-level updates.
        """
        from agex.render.xml import render_events_as_xml

        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Use XML rendering for streaming (instead of structured outputs)
        max_tokens = request_kwargs.get("max_tokens", 4096)
        messages_dicts = render_events_as_xml(events, self._model, max_tokens)

        # Add system message with XML format instructions
        system_with_format = f"{system}\n\n{XML_FORMAT_PRIMER}"

        # Convert to Gemini format
        gemini_messages = self._convert_messages_to_gemini_format(
            system_with_format, messages_dicts
        )

        try:
            # Use streaming API
            response = self.client.generate_content(
                cast(Any, gemini_messages),
                stream=True,
                **request_kwargs,
            )

            # Generator for raw text chunks from Gemini
            def raw_chunks() -> Iterator[str]:
                for chunk in response:
                    if chunk.text:
                        yield chunk.text

            # Parse XML stream into TokenChunks
            yield from tokenize_xml_stream(raw_chunks())

        except Exception as e:
            raise RuntimeError(f"Gemini streaming completion failed: {e}") from e

    def summarize(self, system: str, content: str | List[Event], **kwargs) -> str:
        """Send a summarization request to Gemini (text or events with multimodal)."""
        request_kwargs = {**self._kwargs, **kwargs}

        # Prepare content (text or events)
        is_multimodal, processed = self._prepare_summarization_content(content)

        if is_multimodal:
            # processed is messages list from events
            # Use the existing converter that handles multimodal content
            gemini_messages = self._convert_messages_to_gemini_format(system, processed)
        else:
            # processed is plain text
            gemini_messages = [
                {
                    "role": "user",
                    "parts": [{"text": f"System: {system}\n\n{processed}"}],
                }
            ]

        try:
            response = self.client.generate_content(
                cast(Any, gemini_messages), **request_kwargs
            )
            return response.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini summarization failed: {e}") from e

    def _convert_messages_to_gemini_format(
        self, system: str, messages_dicts: List[dict]
    ) -> List[dict]:
        """
        Convert generic message dicts to Gemini's expected format.

        Note: All images are converted to PNG format by the rendering layer
        (StreamRenderer._serialize_image_to_base64) before reaching this function.
        """
        gemini_messages = []
        system_prepended = False

        for message_dict in messages_dicts:
            role = "user" if message_dict["role"] == "user" else "model"
            parts = []

            # Prepend system content to the first user message
            if role == "user" and not system_prepended:
                parts.append({"text": f"System: {system}"})
                system_prepended = True

            # Process message content
            content = message_dict["content"]
            if isinstance(content, list):
                # Multimodal message
                for part in content:
                    if part["type"] == "text":
                        parts.append({"text": part["text"]})
                    elif part["type"] == "image":
                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": part["image_data"],
                                }
                            }
                        )
            else:
                # Text message
                parts.append({"text": content})

            gemini_messages.append({"role": role, "parts": parts})

        return gemini_messages

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Google Gemini"
