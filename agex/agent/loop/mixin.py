"""
TaskLoopMixin that combines sync and async loop functionality.

This module provides the main mixin class that agents inherit from,
combining shared helper methods with both sync and async task loop implementations.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from datetime import datetime, timezone
from typing import Any

from agex.agent.base import BaseAgent
from agex.agent.primer_text import BUILTIN_PRIMER
from agex.agent.utils import call_sync_or_async
from agex.eval.functions import UserFunction
from agex.render.definitions import render_definitions

from .async_loop import AsyncLoopMixin
from .common import (
    ErrorEvent,
    LLMFail,
    LLMResponse,
    ResponseParseError,
    StreamToken,
    add_event_to_log,
    create_transient_event,
)
from .sync_loop import SyncLoopMixin


class TaskLoopMixin(SyncLoopMixin, AsyncLoopMixin, BaseAgent):
    """
    Mixin that provides the complete task loop implementation.

    Combines:
    - SyncLoopMixin: _task_loop_generator, _run_task_loop
    - AsyncLoopMixin: _atask_loop_generator, _arun_task_loop
    - Shared methods: message building, LLM response handling
    """

    @staticmethod
    def _strip_markdown_code_fence(code: str) -> str:
        """
        Remove surrounding ```python ... ``` (or generic ``` ... ```) fences if the entire
        response code is wrapped in a single fenced block.
        """
        if not isinstance(code, str):
            return code

        text = code.strip()
        if not text.startswith("```"):
            return code

        pattern = r"^```[A-Za-z0-9_+-]*\s*\n([\s\S]*?)\n```\s*$"
        match = re.match(pattern, text)
        if match:
            return match.group(1)
        return code

    def _build_system_message(self) -> str:
        """Build the system message with builtin primer, capabilities primer (or registrations), and agent primer."""
        parts = []

        if self.agex_primer_override is not None:
            parts.append(self.agex_primer_override)
        else:
            parts.append(BUILTIN_PRIMER)

        cap_text = self.capabilities_primer
        if cap_text is not None:
            if cap_text.strip():
                parts.append("# Capabilities Primer\n\n" + cap_text)
        else:
            registered_definitions = render_definitions(self)
            if registered_definitions.strip():
                parts.append("# Registered Resources\n\n" + registered_definitions)

        if self.primer:
            parts.append(self.primer)

        return "\n\n".join(parts)

    def _build_task_message(
        self,
        docstring: str | None,
        inputs_dataclass: type,
        inputs_instance: Any,
        return_type: type,
    ) -> str:
        """Build the initial user message with task description."""
        from agex.agent.task_messages import build_task_message

        return build_task_message(
            docstring, inputs_dataclass, inputs_instance, return_type
        )

    def _get_forefront_message(self, iteration: int, exec_state) -> str | None:
        """
        Get a transient 'forefront' message to be injected into the LLM context.
        """
        messages = []

        # 1. User Functions (always show if present)
        fn_names = exec_state.get("__sys_user_fn_names__", set())
        if fn_names:
            user_fns = []
            missing_names = set()

            for name in sorted(fn_names):
                obj = exec_state.peek(name)
                if isinstance(obj, UserFunction):
                    try:
                        sig = str(obj.__signature__)
                    except Exception:
                        sig = "(...)"

                    doc = inspect.getdoc(obj) or ""
                    if len(doc) > 100:
                        doc = doc[:97] + "..."

                    user_fns.append(f"- {name}{sig}: {doc}")
                else:
                    missing_names.add(name)

            if missing_names:
                new_names = fn_names - missing_names
                exec_state.set("__sys_user_fn_names__", new_names)

            if user_fns:
                messages.append(
                    "## User Defined Functions\n"
                    "The following functions are ALREADY DEFINED in your global scope.\n"
                    "**GUARANTEE**: These functions are LIVE in memory and GUARANTEED to work.\n"
                    "**PERFORMANCE**: Reuse them to reduce token usage and speed up execution.\n"
                    "**DO NOT** redefine them.\n" + "\n".join(user_fns)
                )

        # 2. Iteration Warnings (conditional)
        threshold_idx = int(self.max_iterations * 0.8)
        if self.max_iterations < 10:
            threshold_idx = max(0, self.max_iterations - 3)

        if iteration >= threshold_idx:
            messages.append(
                f"System Note: You are on iteration {iteration + 1} of {self.max_iterations}. Please wrap up."
            )

        if not messages:
            return None

        return "\n\n".join(messages)

    def _get_llm_response(
        self,
        system_message,
        events,
        exec_state,
        on_event,
        on_token,
        transient_message: str | None = None,
    ):
        """Get structured response with retry; emit ErrorEvent per attempt."""
        max_retries = max(0, self.llm_max_retries)
        backoff = 0.5  # Fixed backoff in seconds, doubles on each retry
        provider = self.llm_client.provider_name
        model = self.llm_client.model

        use_streaming = on_token is not None

        messages_to_send = list(events)
        if transient_message:
            transient_event = create_transient_event(
                transient_message,
                messages_to_send[-1].timestamp if messages_to_send else None,
            )
            messages_to_send.append(transient_event)

        attempt = 0
        while True:
            try:
                if use_streaming:
                    title_parts = []
                    thinking_parts = []
                    code_parts = []
                    seen_sections: dict[str, bool] = {
                        "title": False,
                        "thinking": False,
                        "python": False,
                    }

                    for token in self.llm_client.complete_stream(
                        system_message, messages_to_send
                    ):
                        start_flag = (
                            not token.done
                            and token.type in seen_sections
                            and not seen_sections[token.type]
                        )
                        if start_flag and token.type in seen_sections:
                            seen_sections[token.type] = True

                        enriched = StreamToken(
                            type=token.type,
                            content=token.content,
                            done=token.done,
                            agent_name=self.name,
                            full_namespace=getattr(exec_state, "namespace", self.name),
                            timestamp=datetime.now(timezone.utc),
                            start=start_flag,
                        )

                        if on_token is not None:
                            try:
                                on_token(enriched)
                            except Exception:
                                pass

                        if token.type == "title" and not token.done:
                            title_parts.append(token.content)
                        elif token.type == "thinking" and not token.done:
                            thinking_parts.append(token.content)
                        elif token.type == "python" and not token.done:
                            code_parts.append(token.content)

                    return LLMResponse(
                        title="".join(title_parts).strip(),
                        thinking="".join(thinking_parts),
                        code="".join(code_parts),
                    )
                else:
                    return self.llm_client.complete(system_message, messages_to_send)

            except (ResponseParseError, RuntimeError) as e:
                is_last = attempt >= max_retries
                err = ErrorEvent(
                    agent_name=self.name,
                    error=e,
                    recoverable=not is_last,
                )
                add_event_to_log(exec_state, err, on_event=on_event)
                if is_last:
                    raise LLMFail(
                        message=str(e), provider=provider, model=model, retries=attempt
                    )
                sleep_secs = backoff * (2**attempt)
                time.sleep(sleep_secs)
                attempt += 1

    async def _aget_llm_response(
        self,
        system_message,
        events,
        exec_state,
        on_event,
        on_token,
        transient_message: str | None = None,
    ):
        """Async version of _get_llm_response."""
        max_retries = max(0, self.llm_max_retries)
        backoff = 0.5  # Fixed backoff in seconds, doubles on each retry
        provider = self.llm_client.provider_name
        model = self.llm_client.model

        use_streaming = on_token is not None

        messages_to_send = list(events)
        if transient_message:
            transient_event = create_transient_event(
                transient_message,
                messages_to_send[-1].timestamp if messages_to_send else None,
            )
            messages_to_send.append(transient_event)

        attempt = 0
        while True:
            try:
                if use_streaming:
                    title_parts = []
                    thinking_parts = []
                    code_parts = []
                    seen_sections: dict[str, bool] = {
                        "title": False,
                        "thinking": False,
                        "python": False,
                    }

                    async for token in self.llm_client.acomplete_stream(
                        system_message, messages_to_send
                    ):
                        start_flag = (
                            not token.done
                            and token.type in seen_sections
                            and not seen_sections[token.type]
                        )
                        if start_flag and token.type in seen_sections:
                            seen_sections[token.type] = True

                        enriched = StreamToken(
                            type=token.type,
                            content=token.content,
                            done=token.done,
                            agent_name=self.name,
                            full_namespace=getattr(exec_state, "namespace", self.name),
                            timestamp=datetime.now(timezone.utc),
                            start=start_flag,
                        )

                        if on_token is not None:
                            try:
                                res = call_sync_or_async(on_token, enriched)
                                if inspect.isawaitable(res):
                                    await res
                            except Exception:
                                pass

                        if token.type == "title" and not token.done:
                            title_parts.append(token.content)
                        elif token.type == "thinking" and not token.done:
                            thinking_parts.append(token.content)
                        elif token.type == "python" and not token.done:
                            code_parts.append(token.content)

                    return LLMResponse(
                        title="".join(title_parts).strip(),
                        thinking="".join(thinking_parts),
                        code="".join(code_parts),
                    )
                else:
                    return await self.llm_client.acomplete(
                        system_message, messages_to_send
                    )

            except (ResponseParseError, RuntimeError) as e:
                is_last = attempt >= max_retries
                err = ErrorEvent(
                    agent_name=self.name,
                    error=e,
                    recoverable=not is_last,
                )
                add_event_to_log(exec_state, err, on_event=None)
                if on_event:
                    try:
                        res = call_sync_or_async(on_event, err)
                        if inspect.isawaitable(res):
                            await res
                    except Exception:
                        pass

                if is_last:
                    raise LLMFail(
                        message=str(e), provider=provider, model=model, retries=attempt
                    )
                sleep_secs = backoff * (2**attempt)
                await asyncio.sleep(sleep_secs)
                attempt += 1
