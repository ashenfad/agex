"""
Dummy LLM for testing purposes.

This module provides a mock LLM that returns predefined LLMResponse objects
sequentially, useful for testing agent behavior without actual LLM calls.
"""

from typing import AsyncIterator, Iterator, List

from agex.agent.events import Event

from .core import LLM, LLMResponse, TokenChunk


class Dummy(LLM):
    """
    A dummy LLM that returns predefined LLMResponse objects in sequence.
    Useful for testing agent logic without actual LLM calls.
    """

    def __init__(
        self, responses: list[LLMResponse | Exception] | None = None, **kwargs
    ):
        """
        Initialize with a sequence of LLMResponse objects to return.

        Args:
            responses: A list of LLMResponse objects to cycle through. If None, a default
                       response is used.
        """
        # Initialize base class timeout
        self._timeout_seconds = kwargs.get("timeout_seconds", 60.0)

        if responses:
            self.responses = responses
        else:
            from agex.agent.emissions import PythonEmission

            self.responses = [
                LLMResponse(
                    emissions=[
                        PythonEmission(
                            code="print('Hello from Dummy')",
                            thinking="I will use the provided tools.",
                        )
                    ]
                )
            ]
        self.call_count = 0
        self.all_events: list[list[Event]] = []
        self.all_systems: list[str] = []

        # For testing summarization
        self.summary_response: str | None = None
        self.summary_exception: Exception | None = None

        # Renderer selection
        self.renderer = kwargs.get("renderer", "markdown")

    def dump_config(self) -> dict:
        """Serialize client configuration for transport."""
        # Serialize LLMResponse objects; skip Exceptions (can't serialize)
        serialized_responses = [
            r.model_dump() for r in self.responses if isinstance(r, LLMResponse)
        ]

        return {
            "provider": "dummy",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "responses": serialized_responses,
        }

    @classmethod
    def from_config(cls, config: dict) -> "Dummy":
        """Reconstruct from configuration."""
        responses = None
        if "responses" in config and config["responses"]:
            responses = [LLMResponse.model_validate(r) for r in config["responses"]]

        return cls(
            responses=responses,
            timeout_seconds=config.get("timeout_seconds", 60.0),
        )

    def complete(self, system: str, events: List[Event], **kwargs) -> LLMResponse:
        """
        Return the next LLMResponse in the sequence, cycling through the list.
        Exercises the same rendering path as real clients to catch image serialization issues.
        """
        # Store the received data for test inspection
        self.all_systems.append(system)
        self.all_events.append(events)

        # Exercise the rendering path
        renderer_type = getattr(self, "renderer", "markdown")

        has_unsupported_images = False
        try:
            if renderer_type == "xml":
                from agex.llm.formats.xml import render_events_as_xml

                messages_dicts = render_events_as_xml(events)
            else:
                from agex.render.events import render_events_as_markdown

                messages_dicts = render_events_as_markdown(events)

            # Store rendered messages for inspection
            self.all_rendered_messages = getattr(self, "all_rendered_messages", [])
            self.all_rendered_messages.append(messages_dicts)

            # Check for image export failures in rendered messages
            for msg in messages_dicts:
                content = msg.get("content", "")
                if isinstance(content, str) and (
                    "<unsupported image type:" in content
                    or "<image export failed:" in content
                ):
                    has_unsupported_images = True
                elif isinstance(content, list):
                    # Check multimodal content parts
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if (
                                "<unsupported image type:" in text
                                or "<image export failed:" in text
                            ):
                                has_unsupported_images = True
        except Exception:
            # Silently handle rendering errors in dummy client
            pass

        # Get the next item in the cycle
        item = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        # If the item is an exception, raise it to simulate client failure
        if isinstance(item, Exception):
            raise item
        response = item.model_copy(deep=True)

        # If we detected unsupported images, prepend a note on the first
        # thinking-carrying emission so the agent sees the hint.
        if has_unsupported_images:
            from agex.agent.emissions import (
                PythonEmission,
                TerminalEmission,
                ThinkingEmission,
            )

            note = "[Dummy client detected unsupported image type during rendering.]"
            annotated = False
            for em in response.emissions:
                if isinstance(em, (PythonEmission, TerminalEmission)):
                    em.thinking = f"{note}\n{em.thinking or ''}"
                    annotated = True
                    break
                if isinstance(em, ThinkingEmission):
                    em.text = f"{note}\n{em.text}"
                    annotated = True
                    break
            if not annotated:
                response.emissions.insert(0, ThinkingEmission(text=note))

        return response

    async def acomplete(
        self, system: str, events: List[Event], **kwargs
    ) -> LLMResponse:
        """Async version of complete."""
        return self.complete(system, events, **kwargs)

    def complete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> Iterator[TokenChunk]:
        """Stream the response as TokenChunks."""
        from .core import _emissions_to_tokens

        response = self.complete(system, events, **kwargs)
        yield from _emissions_to_tokens(response)

    async def acomplete_stream(
        self, system: str, events: List[Event], **kwargs
    ) -> AsyncIterator[TokenChunk]:
        """Stream the response as Tokens."""
        for token in self.complete_stream(system, events, **kwargs):
            yield token

    def summarize(self, system: str, content: str | List[Event], **kwargs) -> str:
        """Return a deterministic plain text for testing."""
        # Check for configured exception
        if self.summary_exception is not None:
            raise self.summary_exception

        # Check for configured response
        if self.summary_response is not None:
            return self.summary_response

        # Prepare content (handles both text and events)
        is_multimodal, processed = self._prepare_summarization_content(content)

        # For testing, just return a simple string
        if is_multimodal:
            # processed is messages list - count them
            return f"Summary of {len(processed)} messages"
        else:
            # processed is plain text
            return f"{system} {processed}".strip() or "dummy"

    @property
    def context_window(self) -> int:
        return 8192

    @property
    def model(self) -> str:
        return "dummy"

    @property
    def provider_name(self) -> str:
        return "Dummy"
