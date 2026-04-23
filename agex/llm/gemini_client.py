import logging
from typing import Any, AsyncIterator, Iterator, List

from google import genai
from google.genai import types

from agex.agent.events import Event
from agex.llm.core import (
    LLM,
    TokenChunk,
)
from agex.llm.formats import ToolUseWireFormat, WireFormat
from agex.llm.formats.tool_use.gemini_adapter import (
    atranslate_gemini_stream_to_events,
    schemas_to_gemini_function_declarations,
    translate_gemini_stream_to_events,
    translate_messages_to_gemini,
)
from agex.llm.formats.xml import TAG_TITLE

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


class Gemini(LLM):
    """Google Gemini LLM provider (google-genai SDK) with structured outputs."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        google_search: bool = False,
        url_context: bool = False,
        timeout_seconds: float = 90.0,
        wire_format: WireFormat | None = None,
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
        # Gemini 3 delivers signed thought parts natively — the tool-
        # use adapter captures them as ThinkingEmission and round-trips
        # them on replay.  Default the wire format to ``native_thinking
        # =True`` so we stop asking the model to narrate in the schema.
        self._wire_format: WireFormat = wire_format or ToolUseWireFormat(
            native_thinking=True
        )

        # Wire timeout and disable SDK retries (agex handles retries)
        client_kwargs["http_options"] = types.HttpOptions(
            timeout=int(timeout_seconds * 1000),
        )

        # Initialize the unified Client.
        # Supports both API Key (AI Studio) and Vertex AI via explicit kwargs or environment variables.
        self.client = genai.Client(**client_kwargs)

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds for each API call."""
        return self._timeout_seconds

    def dump_config(self) -> dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model,
            "google_search": self._google_search,
            "url_context": self._url_context,
            "timeout_seconds": self.timeout_seconds,
            **self._kwargs,
        }

    def _grounding_tools(self) -> list:
        """Tools required for Gemini's built-in grounding features,
        independent of any agex function declarations."""
        tools: list = []
        if self._google_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if self._url_context:
            tools.append({"url_context": {}})
        return tools

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """Stream tokens from Gemini.

        Dispatches on ``wire_format.tool_schema()``:

        - ``None`` → text-stream path (XML-in-text formats).
        - non-None → provider-native tool-calling path.
        """
        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" in request_kwargs:
            request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

        messages_dicts = self._wire_format.render_events(events)
        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        grounding_primer = _get_grounding_primer(self._google_search, self._url_context)
        if grounding_primer:
            system_with_format = f"{grounding_primer}\n\n{system_with_format}"

        tool_schemas = self._wire_format.tool_schema()
        if tool_schemas is None:
            yield from self._stream_text(
                messages_dicts, system_with_format, request_kwargs
            )
        else:
            yield from self._stream_tools(
                messages_dicts, system_with_format, request_kwargs, tool_schemas
            )

    def _stream_text(
        self,
        messages_dicts: list[dict],
        system_with_format: str,
        request_kwargs: dict,
    ) -> Iterator[TokenChunk]:
        gemini_contents = self._convert_messages_to_gemini_format(messages_dicts)

        # Pre-fill response (only if not grounding, as pre-fill can
        # suppress grounding tools).
        prefill_text = f"<{TAG_TITLE}>"
        if not self._google_search and not self._url_context:
            gemini_contents.append(
                types.Content(role="model", parts=[types.Part(text=prefill_text)])
            )

        tools = self._grounding_tools()

        config = types.GenerateContentConfig(
            system_instruction=system_with_format,
            tools=tools if tools else None,
            **request_kwargs,
        )

        response_stream = self.client.models.generate_content_stream(
            model=self._model,
            contents=gemini_contents,
            config=config,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        def raw_chunks() -> Iterator[Any]:
            if not self._google_search and not self._url_context:
                yield prefill_text

            for chunk in response_stream:
                if chunk.usage_metadata is not None:
                    usage_holder["input_tokens"] = (
                        chunk.usage_metadata.prompt_token_count
                    )
                    usage_holder["output_tokens"] = (
                        chunk.usage_metadata.candidates_token_count
                    )
                text = chunk.text or ""
                yield text

        yield from self._wire_format.parse_text_stream(raw_chunks())

        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

    def _stream_tools(
        self,
        messages_dicts: list[dict],
        system_with_format: str,
        request_kwargs: dict,
        tool_schemas: list[dict],
    ) -> Iterator[TokenChunk]:
        gemini_contents = translate_messages_to_gemini(messages_dicts)

        tools = self._grounding_tools()
        tools.append(
            types.Tool(
                function_declarations=schemas_to_gemini_function_declarations(
                    tool_schemas
                )
            )
        )

        # Surface signed thought parts so the adapter can capture
        # thought_signatures — Gemini 3 requires them to be replayed
        # at the same position on subsequent turns.  User-supplied
        # ``thinking_config`` takes precedence if present.
        if "thinking_config" not in request_kwargs:
            request_kwargs = {
                **request_kwargs,
                "thinking_config": types.ThinkingConfig(include_thoughts=True),
            }

        config = types.GenerateContentConfig(
            system_instruction=system_with_format,
            tools=tools,
            **request_kwargs,
        )

        response_stream = self.client.models.generate_content_stream(
            model=self._model,
            contents=gemini_contents,
            config=config,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        tool_events = translate_gemini_stream_to_events(
            iter(response_stream), usage_holder=usage_holder
        )
        yield from self._wire_format.parse_tool_stream(tool_events)

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
        """Async version of :meth:`complete_stream`."""
        request_kwargs = {**self._kwargs, **kwargs}
        if "max_tokens" in request_kwargs:
            request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

        messages_dicts = self._wire_format.render_events(events)
        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        grounding_primer = _get_grounding_primer(self._google_search, self._url_context)
        if grounding_primer:
            system_with_format = f"{grounding_primer}\n\n{system_with_format}"

        tool_schemas = self._wire_format.tool_schema()
        if tool_schemas is None:
            async for t in self._astream_text(
                messages_dicts, system_with_format, request_kwargs
            ):
                yield t
        else:
            async for t in self._astream_tools(
                messages_dicts, system_with_format, request_kwargs, tool_schemas
            ):
                yield t

    async def _astream_text(
        self,
        messages_dicts: list[dict],
        system_with_format: str,
        request_kwargs: dict,
    ) -> AsyncIterator[TokenChunk]:
        gemini_contents = self._convert_messages_to_gemini_format(messages_dicts)

        prefill_text = f"<{TAG_TITLE}>"
        if not self._google_search and not self._url_context:
            gemini_contents.append(
                types.Content(role="model", parts=[types.Part(text=prefill_text)])
            )

        tools = self._grounding_tools()

        config = types.GenerateContentConfig(
            system_instruction=system_with_format,
            tools=tools if tools else None,
            **request_kwargs,
        )

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self._model,
            contents=gemini_contents,
            config=config,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        async def raw_chunks():
            if not self._google_search and not self._url_context:
                yield prefill_text
            async for chunk in response_stream:
                if chunk.usage_metadata is not None:
                    usage_holder["input_tokens"] = (
                        chunk.usage_metadata.prompt_token_count
                    )
                    usage_holder["output_tokens"] = (
                        chunk.usage_metadata.candidates_token_count
                    )
                yield chunk.text or ""

        async for token in self._wire_format.aparse_text_stream(raw_chunks()):
            yield token

        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

    async def _astream_tools(
        self,
        messages_dicts: list[dict],
        system_with_format: str,
        request_kwargs: dict,
        tool_schemas: list[dict],
    ) -> AsyncIterator[TokenChunk]:
        gemini_contents = translate_messages_to_gemini(messages_dicts)

        tools = self._grounding_tools()
        tools.append(
            types.Tool(
                function_declarations=schemas_to_gemini_function_declarations(
                    tool_schemas
                )
            )
        )

        # Surface signed thought parts so the adapter can capture
        # thought_signatures — Gemini 3 requires them to be replayed
        # at the same position on subsequent turns.  User-supplied
        # ``thinking_config`` takes precedence if present.
        if "thinking_config" not in request_kwargs:
            request_kwargs = {
                **request_kwargs,
                "thinking_config": types.ThinkingConfig(include_thoughts=True),
            }

        config = types.GenerateContentConfig(
            system_instruction=system_with_format,
            tools=tools,
            **request_kwargs,
        )

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self._model,
            contents=gemini_contents,
            config=config,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        tool_events = atranslate_gemini_stream_to_events(
            response_stream, usage_holder=usage_holder
        )
        async for token in self._wire_format.aparse_tool_stream(tool_events):
            yield token

        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

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

        config = types.GenerateContentConfig(
            system_instruction=system, **request_kwargs
        )
        response = self.client.models.generate_content(
            model=self._model,
            contents=gemini_contents,
            config=config,
        )
        return response.text or ""

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
