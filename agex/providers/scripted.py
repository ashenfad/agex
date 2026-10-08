"""A provider that replies from a script, for tests.

The script is nontainer's conformance format: one
``nontainer.conformance.corpus.ModelStep`` per model call (text,
reasoning, tool calls, or a failure). Give a list of steps, or a
callable that returns the next one (a conformance runner's
``Clock.next``, which fires the scenario's outside events between
replies).

It is a :class:`~agex.providers.pydanticai.PydanticAIProvider` over
pydantic-ai's ``FunctionModel``, so the request and the reply go
through the same mapping and streaming a real model's do. Tool calls
get ids of their own (``call_1``, ``call_2``, …), as a provider's would.
A step's ``input_tokens`` is reported as the reply's usage, and a
callable that takes an argument is passed the text of each message the
request carried (as ``Clock.next`` takes it).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any

from nontainer.conformance.corpus import ModelStep
from nontainer.conformance.runner import EXHAUSTED
from pydantic_ai import messages as pai
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingPart,
    DeltaToolCall,
    FunctionModel,
)

from ..record import Message, Usage
from . import ProviderEvent, Reply, Settings, ToolSpec
from .pydanticai import PydanticAIProvider

__all__ = ["ScriptedProvider"]

MODEL_NAME = "scripted"


class ScriptedProvider(PydanticAIProvider):
    """Replies from a script of model steps; ``seen`` holds what each
    request sent, as pydantic-ai messages.

    A step with ``fail="provider"`` raises pydantic-ai's
    ``ModelHTTPError`` (status 529, an overloaded provider), and one
    with ``fail="error"`` a ``RuntimeError``. A list that runs out
    replies with the conformance runner's exhausted step, so a loop that
    asks once too often ends rather than hangs.
    """

    def __init__(self, steps: Iterable[ModelStep] | Callable[..., ModelStep]) -> None:
        if callable(steps):
            takes = bool(inspect.signature(steps).parameters)
            self._next: Callable[[list[str]], ModelStep] = (
                steps if takes else lambda sent: steps()
            )
        else:
            queue = list(steps)
            self._next = lambda sent: queue.pop(0) if queue else EXHAUSTED
        self.seen: list[list[pai.ModelMessage]] = []
        self._calls = 0
        self._reported = 0
        super().__init__(
            FunctionModel(stream_function=self._reply, model_name=MODEL_NAME)
        )

    async def _reply(
        self, messages: list[pai.ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        self.seen.append(list(messages))
        step = self._next(_texts(messages))
        self._reported = step.input_tokens
        if step.fail == "provider":
            raise ModelHTTPError(
                status_code=529, model_name=MODEL_NAME, body="the script is overloaded"
            )
        if step.fail == "error":
            raise RuntimeError("the scripted model failed")
        if step.thinking:
            yield {0: DeltaThinkingPart(content=step.thinking)}
        if step.text:
            yield step.text
        for index, call in enumerate(step.tool_calls, start=1):
            self._calls += 1
            yield {
                index: DeltaToolCall(
                    name=call.name,
                    json_args=json.dumps(call.args),
                    tool_call_id=f"call_{self._calls}",
                )
            }

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        settings: Settings = Settings(),
    ) -> AsyncIterator[ProviderEvent]:
        self._reported = 0
        async for event in super().stream(messages, tools, settings):
            if isinstance(event, Reply) and self._reported:
                usage = replace(
                    event.message.usage or Usage(), input_tokens=self._reported
                )
                event = Reply(message=replace(event.message, usage=usage))
            yield event


def _texts(messages: Sequence[pai.ModelMessage]) -> list[str]:
    """The text of each part a request carried, in order."""
    out = []
    for message in messages:
        for part in message.parts:
            content = getattr(part, "content", None)
            if isinstance(content, str) and content:
                out.append(content)
            elif content is not None and not isinstance(content, str):
                out.append(str(content))
    return out
