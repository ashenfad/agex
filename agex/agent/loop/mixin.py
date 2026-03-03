"""
TaskLoopMixin that combines sync and async loop functionality.

This module provides the main mixin class that agents inherit from,
combining shared helper methods with both sync and async task loop implementations.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from typing import Any

from agex.agent.base import BaseAgent
from agex.agent.chapter import (
    Chapter,
    build_numbered_event_index,
    get_latest_input_tokens,
    should_trigger_chaptering,
)
from agex.agent.events import ChapterEvent
from agex.agent.primer_text import BUILTIN_PRIMER
from agex.agent.utils import call_sync_or_async
from agex.eval.analysis import get_workspace_recap
from agex.render.definitions import render_definitions
from agex.state.log import get_events_from_log, replace_events_with_chapters

from .async_loop import AsyncLoopMixin
from .common import (
    ErrorEvent,
    LLMFail,
    ResponseBuilder,
    ResponseParseError,
    add_event_to_log,
    create_transient_event,
)
from .sync_loop import SyncLoopMixin


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Build tuple of retryable exception types from available SDK packages."""
    retryable: list[type[Exception]] = [ResponseParseError]
    try:
        import anthropic

        retryable.extend(
            [
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
            ]
        )
    except ImportError:
        pass
    try:
        import openai

        retryable.extend(
            [
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            ]
        )
    except ImportError:
        pass
    try:
        from google.genai import errors as genai_errors

        retryable.append(genai_errors.ServerError)
    except ImportError:
        pass
    return tuple(retryable)


_RETRYABLE = _retryable_exceptions()


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
                obj = exec_state.get(name)
                if obj is not None and callable(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except (ValueError, TypeError):
                        sig = "(...)"

                    doc = inspect.getdoc(obj) or ""
                    if len(doc) > 100:
                        doc = doc[:97] + "..."

                    user_fns.append(f"- {name}{sig}: {doc}")
                else:
                    missing_names.add(name)

            if missing_names:
                new_names = fn_names - missing_names
                exec_state["__sys_user_fn_names__"] = new_names

            if user_fns:
                messages.append(
                    "## User Defined Functions\n"
                    "The following functions are ALREADY DEFINED in your global scope.\n"
                    "**GUARANTEE**: These functions are LIVE in memory and GUARANTEED to work.\n"
                    "**PERFORMANCE**: Reuse them to reduce token usage and speed up execution.\n"
                    "**DO NOT** redefine them.\n" + "\n".join(user_fns)
                )

        # 2. Workspace Recap (Inventory)
        # We assume the session matches what's in exec_state (usually 'default' unless specified)
        # We use getattr to safely access session if it's stored in state
        session = getattr(exec_state, "session", "default")
        recap = get_workspace_recap(self, session=session)
        if recap:
            messages.append(
                "## Workspace Module Inventory\n"
                "The following modules are available in your virtual filesystem.\n"
                "**IMPORTANT**: To use them, you must `import` them first.\n" + recap
            )

        # 3. Iteration Warnings (conditional)
        threshold_idx = int(self.max_iterations * 0.8)
        if self.max_iterations < 10:
            threshold_idx = max(0, self.max_iterations - 3)

        if iteration >= threshold_idx:
            messages.append(
                f"System Note: You are on iteration {iteration + 1} of {self.max_iterations}. Please wrap up."
            )

        # 4. Current Working Directory (show if not at root)
        try:
            fs = self.fs()
            if fs is not None:
                cwd = fs.getcwd()
                if cwd and cwd != "/":
                    messages.append(f"**Current Directory**: `{cwd}`")
        except Exception:
            pass  # FS not configured or error - skip

        if not messages:
            return None

        return "\n\n".join(messages)

    def _maybe_chapter(self, state, session, on_event, on_token):
        """Run chapter task if context exceeds high water mark.

        Checks the most recent ActionEvent's input_tokens against
        log_high_water_tokens. If triggered, calls the __chapter__ task
        which returns Chapter instances. Converts those to ChapterEvents
        and applies them to the event log.

        Loops up to 3 rounds or until below low water mark.
        """
        if self._chapter_task is None:
            return

        logger = logging.getLogger("agex.chapters")

        for round_num in range(3):  # Max 3 chaptering rounds
            all_events = get_events_from_log(state)
            visible_events = [e for e in all_events if not isinstance(e, ErrorEvent)]

            if not should_trigger_chaptering(all_events, self.log_high_water_tokens):
                return

            # Build event index and call chapter task
            index_text = build_numbered_event_index(visible_events)
            try:
                chapters = self._chapter_task(
                    event_index=index_text,
                    session=session,
                    on_event=on_event,
                    on_token=on_token,
                )
            except Exception:
                logger.debug(
                    "Chapter task failed (round %d)", round_num + 1, exc_info=True
                )
                return

            if not chapters:
                logger.debug("Agent returned no chapters (round %d)", round_num + 1)
                return

            # Validate and convert to ChapterEvents
            # Build mapping from 1-based visible index to 0-based event log index
            visible_to_log = []
            for log_idx, event in enumerate(all_events):
                if not isinstance(event, ErrorEvent):
                    visible_to_log.append(log_idx)

            chapters_and_ranges = []
            for ch in chapters:
                if not isinstance(ch, Chapter):
                    logger.debug("Skipping non-Chapter object: %s", type(ch).__name__)
                    continue
                # Convert 1-based inclusive to 0-based exclusive using visible mapping
                if ch.start < 1 or ch.end < ch.start:
                    logger.debug(
                        "Skipping invalid range: start=%d end=%d", ch.start, ch.end
                    )
                    continue
                if ch.start > len(visible_to_log) or ch.end > len(visible_to_log):
                    logger.debug(
                        "Skipping out-of-bounds range: start=%d end=%d (max=%d)",
                        ch.start,
                        ch.end,
                        len(visible_to_log),
                    )
                    continue

                log_start = visible_to_log[ch.start - 1]
                log_end = visible_to_log[ch.end - 1] + 1  # exclusive

                # Collect the events being chaptered
                chaptered_events = list(all_events[log_start:log_end])

                chapter_event = ChapterEvent(
                    agent_name=self.name,
                    name=ch.name,
                    message=ch.message,
                    events=chaptered_events,
                )
                chapters_and_ranges.append((log_start, log_end, chapter_event))

            if not chapters_and_ranges:
                return

            try:
                replace_events_with_chapters(state, chapters_and_ranges)
            except ValueError:
                logger.debug(
                    "Failed to apply chapters (round %d)",
                    round_num + 1,
                    exc_info=True,
                )
                return

            logger.debug(
                "Applied %d chapter(s) in round %d",
                len(chapters_and_ranges),
                round_num + 1,
            )

            # Check low water mark — stop if we've compacted enough
            if self.log_low_water_tokens is not None:
                updated_events = get_events_from_log(state)
                latest_tokens = get_latest_input_tokens(updated_events)
                if (
                    latest_tokens is not None
                    and latest_tokens <= self.log_low_water_tokens
                ):
                    logger.debug(
                        "Below low water mark (%d <= %d), stopping",
                        latest_tokens,
                        self.log_low_water_tokens,
                    )
                    return

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
        provider = self.llm.provider_name
        model = self.llm.model

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
                    builder = ResponseBuilder(self.name, exec_state)
                    for token in self.llm.complete_stream(
                        system_message, messages_to_send
                    ):
                        enriched = builder.process_token(token)
                        if on_token is not None:
                            try:
                                on_token(enriched)
                            except Exception:
                                pass
                    return builder.build()
                else:
                    return self.llm.complete(system_message, messages_to_send)

            except _RETRYABLE as e:
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
        provider = self.llm.provider_name
        model = self.llm.model

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
                    builder = ResponseBuilder(self.name, exec_state)
                    async for token in self.llm.acomplete_stream(
                        system_message, messages_to_send
                    ):
                        enriched = builder.process_token(token)
                        if on_token is not None:
                            try:
                                res = call_sync_or_async(on_token, enriched)
                                if inspect.isawaitable(res):
                                    await res
                            except Exception:
                                pass
                    return builder.build()
                else:
                    return await self.llm.acomplete(system_message, messages_to_send)

            except _RETRYABLE as e:
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
