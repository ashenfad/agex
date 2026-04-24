from typing import Any, AsyncIterator, Iterator, List

import openai

from agex.agent.events import Event
from agex.llm.core import (
    LLM,
    TokenChunk,
)
from agex.llm.formats import ToolUseWireFormat, WireFormat
from agex.llm.formats.tool_use.openai_adapter import (
    atranslate_openai_stream_to_events,
    schemas_to_openai_tools,
    translate_messages_to_openai,
    translate_openai_stream_to_events,
)
from agex.llm.formats.tool_use.openai_responses_adapter import (
    atranslate_openai_responses_stream_to_events,
    schemas_to_openai_responses_tools,
    translate_messages_to_openai_responses,
    translate_openai_responses_stream_to_events,
)
from agex.tokenizers import get_tokenizer

# Define keys for client setup vs. completion
CLIENT_CONFIG_KEYS = {"api_key", "base_url", "organization", "timeout", "max_retries"}

# Default reasoning_effort.  GPT-5+ agex clients are assumed to run a
# reasoning model; the API's own default is "none" which produces the
# wrong behaviour for agentic multi-step tasks.  Users on a non-
# reasoning model should override with ``reasoning_effort=None`` in
# kwargs or pass their preferred value.
_DEFAULT_REASONING_EFFORT = "low"


def _is_reasoning_model(model: str) -> bool:
    """Heuristic: models that must use the Responses endpoint.

    GPT-5 family and the o1/o3 reasoning models are Responses-native;
    Chat Completions rejects ``reasoning_effort`` + function tools for
    several of them (e.g. ``gpt-5.4-nano``) and Responses is where
    OpenAI is pushing new features (encrypted reasoning round-trip,
    richer output item shapes).  Everything else (gpt-4-family, 3.5)
    stays on Chat Completions.
    """
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")


def _with_reasoning_default_chat(request_kwargs: dict) -> dict:
    """No-op on the Chat Completions path.

    The dispatch routes every gpt-5* / o1* / o3* to the Responses
    endpoint, so this branch only runs for non-reasoning models
    (gpt-4* etc.) — which reject ``reasoning_effort`` with a 400.
    Callers who've forced a reasoning model onto Chat Completions
    via ``use_responses=False`` can still pass ``reasoning_effort``
    explicitly; we just don't default one.
    """
    return request_kwargs


def _with_reasoning_default_responses(request_kwargs: dict) -> dict:
    """Default the Responses ``reasoning`` block to ``{"effort": "low"}``.

    Also accepts legacy ``reasoning_effort=`` kwargs and folds them
    into the nested shape, so user code that targeted Chat Completions
    keeps working when the model dispatches to Responses.  Setting
    ``reasoning=None`` or ``reasoning_effort=None`` explicitly opts
    out.
    """
    kwargs = {**request_kwargs}
    if "reasoning_effort" in kwargs:
        effort = kwargs.pop("reasoning_effort")
        if "reasoning" not in kwargs and effort is not None:
            kwargs["reasoning"] = {"effort": effort}
        return kwargs
    if "reasoning" in kwargs:
        return kwargs
    return {**kwargs, "reasoning": {"effort": _DEFAULT_REASONING_EFFORT}}


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
        model: str = "gpt-5-mini",
        timeout_seconds: float = 90.0,
        wire_format: WireFormat | None = None,
        use_responses: bool | None = None,
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
        # Pick the endpoint.  Responses is required for GPT-5-family
        # reasoning models (Chat Completions rejects reasoning_effort +
        # function tools for several of them) and is where new features
        # are shipping; Chat Completions still covers gpt-4-family and
        # earlier.  Callers can force either with ``use_responses=``.
        self._use_responses = (
            use_responses if use_responses is not None else _is_reasoning_model(model)
        )
        # GPT-5+ reasons server-side and delivers surfaced text
        # natively; narration-via-schema is redundant.  Default
        # ``native_thinking=True`` on the wire format.  Users on
        # chat-class models who want narrated thinking can pass an
        # explicit ``ToolUseWireFormat()``.
        self._wire_format: WireFormat = wire_format or ToolUseWireFormat(
            native_thinking=True
        )
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
        """Stream tokens from OpenAI via the provider-native tool-use
        path.  Dispatches to Responses for reasoning-class models and
        Chat Completions for everything else.
        """
        request_kwargs = {**self._kwargs, **kwargs}
        messages_dicts = self._wire_format.render_events(events)
        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        full_messages = [
            {"role": "system", "content": system_with_format}
        ] + messages_dicts
        tool_schemas = self._wire_format.tool_schema()
        if self._use_responses:
            yield from self._stream_responses(
                full_messages, request_kwargs, tool_schemas
            )
        else:
            yield from self._stream_chat(full_messages, request_kwargs, tool_schemas)

    # --- Chat Completions path ------------------------------------

    def _stream_chat(
        self,
        full_messages: list[dict],
        request_kwargs: dict,
        tool_schemas: list[dict],
    ) -> Iterator[TokenChunk]:
        translated = translate_messages_to_openai(full_messages)
        tools = schemas_to_openai_tools(tool_schemas)
        request_kwargs = _with_reasoning_default_chat(request_kwargs)

        stream = self.client.chat.completions.create(
            model=self._model,
            messages=[_format_message_for_openai(msg) for msg in translated],  # type: ignore
            tools=tools,  # type: ignore
            tool_choice=request_kwargs.pop("tool_choice", "required"),
            stream=True,
            stream_options={"include_usage": True},
            **request_kwargs,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        tool_events = translate_openai_stream_to_events(
            iter(stream), usage_holder=usage_holder
        )
        yield from self._wire_format.parse_tool_stream(tool_events)

        yield TokenChunk(
            type="thinking",
            content="",
            done=True,
            input_tokens=usage_holder["input_tokens"],
            output_tokens=usage_holder["output_tokens"],
        )

    # --- Responses path -------------------------------------------

    def _stream_responses(
        self,
        full_messages: list[dict],
        request_kwargs: dict,
        tool_schemas: list[dict],
    ) -> Iterator[TokenChunk]:
        input_items = translate_messages_to_openai_responses(full_messages)
        tools = schemas_to_openai_responses_tools(tool_schemas)
        request_kwargs = _with_reasoning_default_responses(request_kwargs)

        # Stateless round-trip: we don't rely on OpenAI's conversation
        # store; each turn sends the full history as input items and
        # replays the prior reasoning items from their encrypted
        # payloads.  ``include=["reasoning.encrypted_content"]`` is the
        # switch that surfaces those payloads in the response.
        kwargs = {**request_kwargs, "store": request_kwargs.get("store", False)}
        include = list(kwargs.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        kwargs["include"] = include

        stream = self.client.responses.create(
            model=self._model,
            input=input_items,  # type: ignore
            tools=tools,  # type: ignore
            tool_choice=kwargs.pop("tool_choice", "required"),
            stream=True,
            **kwargs,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        tool_events = translate_openai_responses_stream_to_events(
            iter(stream), usage_holder=usage_holder
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
        messages_dicts = self._wire_format.render_events(events)
        system_with_format = f"{system}\n\n{self._wire_format.format_primer()}"
        full_messages = [
            {"role": "system", "content": system_with_format}
        ] + messages_dicts
        tool_schemas = self._wire_format.tool_schema()
        if self._use_responses:
            async for t in self._astream_responses(
                full_messages, request_kwargs, tool_schemas
            ):
                yield t
        else:
            async for t in self._astream_chat(
                full_messages, request_kwargs, tool_schemas
            ):
                yield t

    async def _astream_chat(
        self,
        full_messages: list[dict],
        request_kwargs: dict,
        tool_schemas: list[dict],
    ) -> AsyncIterator[TokenChunk]:
        translated = translate_messages_to_openai(full_messages)
        tools = schemas_to_openai_tools(tool_schemas)
        request_kwargs = _with_reasoning_default_chat(request_kwargs)

        stream = await self.async_client.chat.completions.create(
            model=self._model,
            messages=[_format_message_for_openai(msg) for msg in translated],  # type: ignore
            tools=tools,  # type: ignore
            tool_choice=request_kwargs.pop("tool_choice", "required"),
            stream=True,
            stream_options={"include_usage": True},
            **request_kwargs,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        tool_events = atranslate_openai_stream_to_events(
            stream, usage_holder=usage_holder
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

    async def _astream_responses(
        self,
        full_messages: list[dict],
        request_kwargs: dict,
        tool_schemas: list[dict],
    ) -> AsyncIterator[TokenChunk]:
        input_items = translate_messages_to_openai_responses(full_messages)
        tools = schemas_to_openai_responses_tools(tool_schemas)
        request_kwargs = _with_reasoning_default_responses(request_kwargs)

        kwargs = {**request_kwargs, "store": request_kwargs.get("store", False)}
        include = list(kwargs.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        kwargs["include"] = include

        stream = await self.async_client.responses.create(
            model=self._model,
            input=input_items,  # type: ignore
            tools=tools,  # type: ignore
            tool_choice=kwargs.pop("tool_choice", "required"),
            stream=True,
            **kwargs,
        )

        usage_holder: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

        tool_events = atranslate_openai_responses_stream_to_events(
            stream, usage_holder=usage_holder
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
