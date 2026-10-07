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
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterable
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

    def __init__(self, steps: Iterable[ModelStep] | Callable[[], ModelStep]) -> None:
        if callable(steps):
            self._next: Callable[[], ModelStep] = steps
        else:
            queue = list(steps)
            self._next = lambda: queue.pop(0) if queue else EXHAUSTED
        self.seen: list[list[pai.ModelMessage]] = []
        self._calls = 0
        super().__init__(
            FunctionModel(stream_function=self._reply, model_name=MODEL_NAME)
        )

    async def _reply(
        self, messages: list[pai.ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        self.seen.append(list(messages))
        step = self._next()
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
