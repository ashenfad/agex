import json
import logging
from typing import Any, AsyncIterator, Iterator, List

from google import genai
from google.genai import types

from agex.agent.events import Event
from agex.llm.core import (
    LLMClient,
    LLMResponse,
    TokenChunk,
    with_timeout,
    with_timeout_async,
)
from agex.llm.xml import TAG_TITLE, XML_FORMAT_PRIMER, tokenize_xml_stream

logger = logging.getLogger(__name__)

CLIENT_CONFIG_KEYS = {"api_key", "vertexai"}

GROUNDING_PRIMER_TEMPLATE = """
# Grounding Tools Enabled
You have access to gemini grounding tools. These tools are available external
to agex. If you choose to use them, do so before the <TITLE>.

When using them, please make a detailed summary of what you learn and include it in your
<THINKING> section. This will enable you to remember the summary long-term.
"""


def _get_grounding_primer(google_search: bool, url_context: bool) -> str:
    if not (google_search or url_context):
        return ""
    return GROUNDING_PRIMER_TEMPLATE


class GeminiClient(LLMClient):
    """Client for Google's Gemini API (google-genai SDK) with structured outputs."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        google_search: bool = False,
        url_context: bool = False,
        timeout_seconds: float = 90.0,
        **kwargs,
    ):
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
        self._google_search = google_search
        self._url_context = url_context
        self._timeout_seconds = timeout_seconds

        # Initialize the unified Client.
        # Supports both API Key (AI Studio) and Vertex AI via explicit kwargs or environment variables.
        self.client = genai.Client(**client_kwargs)

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds for each API call."""
        return self._timeout_seconds

    def complete(self, system: str, events: List[Event], **kwargs) -> LLMResponse:
        """
        Send events to Gemini and return a structured response.

        Includes timeout and retry logic to prevent indefinite hangs.
        """
        from agex.render.events import render_events_as_markdown

        # Combine kwargs, giving precedence to method-level ones
        request_kwargs = {**self._kwargs, **kwargs}

        # Remap standard params if needed (google-genai uses config object)
        if "max_tokens" in request_kwargs:
            request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

        # Use rendering helper to convert events to markdown messages
        messages_dicts = render_events_as_markdown(events)

        # Convert to Gemini format
        gemini_contents = self._convert_messages_to_gemini_format(messages_dicts)

        # Define schema for structured output
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

        # Create config
        tools = []
        if self._google_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        if self._url_context:
            # Based on documentation, pass as a dict or dynamic type
            tools.append({"url_context": {}})

        if tools:
            grounding_primer = _get_grounding_primer(
                self._google_search, self._url_context
            )
            if grounding_primer:
                system = f"{grounding_primer}\n\n{system}"

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_schema,
            tools=tools if tools else None,
            **request_kwargs,
        )

        def _make_request():
            return self.client.models.generate_content(
                model=self._model,
                contents=gemini_contents,
                config=config,
            )

        try:
            # Execute with timeout
            response = with_timeout(_make_request, self.timeout_seconds)

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

        except TimeoutError as e:
            raise RuntimeError(f"Gemini completion timed out: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini completion failed: {e}") from e

    async def acomplete(
        self, system: str, events: List[Event], **kwargs
    ) -> LLMResponse:
        """Async version of complete with timeout and retry logic."""
        from agex.render.events import render_events_as_markdown

        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" in request_kwargs:
            request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

        messages_dicts = render_events_as_markdown(events)
        gemini_contents = self._convert_messages_to_gemini_format(messages_dicts)

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

        tools = []
        if self._google_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if self._url_context:
            tools.append({"url_context": {}})
        if tools:
            grounding_primer = _get_grounding_primer(
                self._google_search, self._url_context
            )
            if grounding_primer:
                system = f"{grounding_primer}\n\n{system}"

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_schema,
            tools=tools if tools else None,
            **request_kwargs,
        )

        async def _make_request():
            return await self.client.aio.models.generate_content(
                model=self._model,
                contents=gemini_contents,
                config=config,
            )

        try:
            # Execute with timeout
            response = await with_timeout_async(_make_request, self.timeout_seconds)

            if not response.text:
                raise RuntimeError("Gemini returned empty response")

            try:
                parsed_response = json.loads(response.text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse Gemini JSON response: {e}")

            thinking = parsed_response.get("thinking", "")
            code = parsed_response.get("code", "")
            return LLMResponse(thinking=thinking, code=code)

        except TimeoutError as e:
            raise RuntimeError(f"Gemini completion timed out: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini completion failed: {e}") from e

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """
        Stream tokens from Gemini using XML format.
        """
        from agex.render.xml import render_events_as_xml

        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" in request_kwargs:
            request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

        messages_dicts = render_events_as_xml(events)

        # Add system message with XML format instructions
        system_with_format = f"{system}\n\n{XML_FORMAT_PRIMER}"

        grounding_primer = _get_grounding_primer(self._google_search, self._url_context)
        if grounding_primer:
            system_with_format = f"{grounding_primer}\n\n{system_with_format}"

        # Convert to Gemini format
        gemini_contents = self._convert_messages_to_gemini_format(messages_dicts)

        # Pre-fill response (only if not grounding, as pre-fill can suppress grounding tools)
        prefill_text = f"<{TAG_TITLE}>"
        if not self._google_search and not self._url_context:
            gemini_contents.append(
                types.Content(role="model", parts=[types.Part(text=prefill_text)])
            )

        try:
            tools = []
            if self._google_search:
                tools.append(types.Tool(google_search=types.GoogleSearch()))

            if self._url_context:
                tools.append({"url_context": {}})

            config = types.GenerateContentConfig(
                system_instruction=system_with_format,
                tools=tools if tools else None,
                **request_kwargs,
            )

            # Streaming call
            response_stream = self.client.models.generate_content_stream(
                model=self._model,
                contents=gemini_contents,
                config=config,
            )

            def raw_chunks() -> Iterator[Any]:
                if not self._google_search and not self._url_context:
                    yield prefill_text

                for chunk in response_stream:
                    text = chunk.text or ""
                    yield text

            yield from tokenize_xml_stream(raw_chunks())

        except Exception as e:
            raise RuntimeError(f"Gemini streaming completion failed: {e}") from e

    async def acomplete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """Async version of complete_stream."""
        from agex.llm.xml import atokenize_xml_stream
        from agex.render.xml import render_events_as_xml

        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" in request_kwargs:
            request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

        messages_dicts = render_events_as_xml(events)
        system_with_format = f"{system}\n\n{XML_FORMAT_PRIMER}"

        grounding_primer = _get_grounding_primer(self._google_search, self._url_context)
        if grounding_primer:
            system_with_format = f"{grounding_primer}\n\n{system_with_format}"

        gemini_contents = self._convert_messages_to_gemini_format(messages_dicts)

        prefill_text = f"<{TAG_TITLE}>"
        if not self._google_search and not self._url_context:
            gemini_contents.append(
                types.Content(role="model", parts=[types.Part(text=prefill_text)])
            )

        try:
            tools = []
            if self._google_search:
                tools.append(types.Tool(google_search=types.GoogleSearch()))
            if self._url_context:
                tools.append({"url_context": {}})

            config = types.GenerateContentConfig(
                system_instruction=system_with_format,
                tools=tools if tools else None,
                **request_kwargs,
            )

            # Get stream handle
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self._model,
                contents=gemini_contents,
                config=config,
            )

            async def raw_chunks():
                if not self._google_search and not self._url_context:
                    yield prefill_text
                async for chunk in response_stream:
                    yield chunk.text or ""

            async for token in atokenize_xml_stream(raw_chunks()):
                yield token

        except Exception as e:
            raise RuntimeError(f"Gemini streaming completion failed: {e}") from e

    def summarize(self, system: str, content: str | List[Event], **kwargs) -> str:
        """Send a summarization request to Gemini."""
        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" in request_kwargs:
            request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

        is_multimodal, processed = self._prepare_summarization_content(content)

        if is_multimodal:
            gemini_contents = self._convert_messages_to_gemini_format(processed)
        else:
            gemini_contents = [
                types.Content(role="user", parts=[types.Part(text=str(processed))])
            ]

        try:
            config = types.GenerateContentConfig(
                system_instruction=system, **request_kwargs
            )
            response = self.client.models.generate_content(
                model=self._model,
                contents=gemini_contents,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini summarization failed: {e}") from e

    def _convert_messages_to_gemini_format(
        self, messages_dicts: List[dict]
    ) -> List[types.Content]:
        """
        Convert generic message dicts to Gemini's types.Content objects.
        """
        gemini_contents = []

        for message_dict in messages_dicts:
            role = "user" if message_dict["role"] == "user" else "model"
            parts = []

            content = message_dict["content"]
            if isinstance(content, list):
                # Multimodal
                for part in content:
                    if part["type"] == "text":
                        parts.append(types.Part(text=part["text"]))
                    elif part["type"] == "image":
                        # Updated SDK uses explicit Part types usually, or dicts.
                        # inline_data matches legacy but let's see if types.Part supports it nicely.
                        # types.Part(inline_data=types.Blob(mime_type=..., data=...))
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=part["image_data"],
                                )
                            )
                        )
            else:
                parts.append(types.Part(text=content))

            gemini_contents.append(types.Content(role=role, parts=parts))

        return gemini_contents

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Google Gemini"
