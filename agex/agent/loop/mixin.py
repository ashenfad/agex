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
    build_boundary_index,
    has_completable_boundary,
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
    EmissionsBuilder,
    ErrorEvent,
    LLMFail,
    ResponseParseError,
    add_event_to_log,
    create_transient_event,
)
from .sync_loop import SyncLoopMixin

_RETRYABLE_CACHE: tuple[type[Exception], ...] | None = None


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Tuple of retryable exception types from available SDK packages.

    Resolved lazily on first call — importing the provider SDKs eagerly
    at module load would pull heavy transitive deps (e.g. google-genai
    pulls PIL) into every ``import agex``.
    """
    global _RETRYABLE_CACHE
    if _RETRYABLE_CACHE is not None:
        return _RETRYABLE_CACHE

    retryable: list[type[Exception]] = [ResponseParseError]
    try:
        import anthropic  # noqa: PLC0415

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
        import openai  # noqa: PLC0415

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
        from google.genai import errors as genai_errors  # noqa: PLC0415

        retryable.append(genai_errors.ServerError)
    except ImportError:
        pass

    _RETRYABLE_CACHE = tuple(retryable)
    return _RETRYABLE_CACHE


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

    def _discover_skills(self) -> list[tuple[str, str, list[str]]]:
        """Parse registered skills and return (name, description, modules) tuples.

        The name is already resolved at registration time (YAML frontmatter
        takes priority, then filename/dir fallback). Here we extract
        description and modules from frontmatter.
        """
        if not self._skills:
            return []

        skills = []
        for name, files in self._skills:
            content_bytes = files.get("SKILL.md", b"")
            content = content_bytes.decode("utf-8", errors="replace")
            description = ""
            modules: list[str] = []

            fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if fm_match:
                in_modules = False
                in_description = False
                desc_lines: list[str] = []
                for line in fm_match.group(1).splitlines():
                    stripped = line.strip()
                    if stripped.startswith("description:"):
                        val = stripped[12:].strip().strip("\"'")
                        # Ignore YAML block scalar indicators (|, >)
                        if val in ("|", ">", "|+", "|-", ">+", ">-"):
                            val = ""
                        desc_lines = [val] if val else []
                        in_description = True
                        in_modules = False
                    elif stripped == "modules:":
                        in_description = False
                        in_modules = True
                    elif (
                        in_description
                        and line[0:1] in (" ", "\t")
                        and ":" not in stripped
                    ):
                        desc_lines.append(stripped)
                    elif in_modules and stripped.startswith("- "):
                        modules.append(stripped[2:].strip())
                    elif in_modules and not stripped.startswith("-"):
                        in_modules = False
                    else:
                        in_description = False
                description = " ".join(desc_lines)

            skills.append((name, description, modules))

        skills.sort()
        return skills

    def _build_system_message(self) -> str:
        """Build the system message with builtin primer, capabilities primer (or registrations), and agent primer."""
        parts = []

        if self.agex_primer_override is not None:
            parts.append(self.agex_primer_override)
        else:
            parts.append(BUILTIN_PRIMER)

        # Discover skills early so we can pass module→skill mapping to render_definitions
        skills = self._discover_skills()
        module_to_skill: dict[str, str] = {}
        for skill_name, _desc, skill_modules in skills:
            for mod in skill_modules:
                module_to_skill[mod] = skill_name

        cap_text = self.capabilities_primer
        if cap_text is not None:
            if cap_text.strip():
                parts.append("# Capabilities Primer\n\n" + cap_text)
        else:
            registered_definitions = render_definitions(
                self, module_to_skill=module_to_skill
            )
            if registered_definitions.strip():
                parts.append("# Registered Resources\n\n" + registered_definitions)

        # List available skills
        if skills:
            lines = [
                "# Skills",
                "",
                "Skills provide API docs for registered libraries whose APIs "
                "differ from what you may expect. Always read a skill BEFORE "
                "using its library — do not guess at function signatures or "
                "field names.",
                "  cat /skills/<name>/SKILL.md",
                "",
                "Available skills:",
            ]
            for name, desc, _mods in skills:
                if desc:
                    lines.append(f"- {name}: {desc}")
                else:
                    lines.append(f"- {name}")
            parts.append("\n".join(lines))

        # Permission scopes — included only when the agent declares scoped
        # capabilities. Gated on the *static* scope set (never per-session
        # grant state), so the system prompt stays cache-stable across
        # grants/revokes. Omitted entirely when there are no scopes, so an
        # agent without gated capabilities can't hallucinate requests.
        scope_names = self.scope_names
        if scope_names:
            parts.append(
                "\n".join(
                    [
                        "# Permission Scopes",
                        "",
                        "Some capabilities are gated behind a *scope* and stay "
                        "locked until the user grants it for this session. "
                        "Using a locked capability raises a ScopeRequired error "
                        "naming the scope.",
                        "",
                        "To request a scope, end your turn with:",
                        "  task_request_permission(scope='<name>', reason='<why>')",
                        "",
                        "This suspends the task until the user decides; you then "
                        "resume with their decision. Request proactively when "
                        "you can see a capability you'll need is locked. If a "
                        "request is denied, adapt or fail gracefully — do not "
                        "re-request the same scope.",
                        "",
                        "Declarable scopes: " + ", ".join(sorted(scope_names)),
                    ]
                )
            )

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

        # Workspace Recap (Inventory)
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
            # The renderer wraps SystemNoteEvent messages with a
            # ``[system]`` prefix on the way to the LLM, so the
            # framework attribution is already explicit — no need
            # to repeat it in the body.
            messages.append(
                f"You are on iteration {iteration + 1} of {self.max_iterations}. Please wrap up."
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

    async def _maybe_chapter(self, state, session, on_event, on_token):
        """Run chapter task if context exceeds the chaptering trigger.

        Checks the most recent ActionEvent's input_tokens against
        chaptering_trigger. If triggered, builds the boundary index
        (one entry per non-chapter ``TaskStartEvent`` and per prior
        ``ChapterEvent``) and runs the ``__chapter__`` task. The
        chapter task returns ``Chapter`` instances naming 1-based
        inclusive boundary ranges to fold; each is converted to a
        ``ChapterEvent`` that replaces the underlying log slice.
        """
        if self._chapter_task is None:
            return

        logger = logging.getLogger("agex.chapters")

        all_events = get_events_from_log(state)

        if not should_trigger_chaptering(all_events, self.chaptering_trigger):
            return

        # Build the boundary index. ``ranges`` are 0-based, end-exclusive
        # log slices — each entry covers a single boundary's owned events.
        index_text, ranges = build_boundary_index(all_events)

        # Skip the chapter task when there's nothing safe to fold.
        # The trigger fires *during* the parent's flow, so the parent's
        # in-progress task is one of the boundaries — but its range has
        # no terminator yet and the primer rules out folding ongoing
        # work. We need at least one *completable* boundary (a closed
        # task or a prior ChapterEvent). Without one, invoking the
        # chapter task wastes an LLM call and pollutes the parent's
        # log with empty-result bookkeeping.
        if not has_completable_boundary(all_events, ranges):
            logger.debug("No completable boundary; skipping chapter task")
            return

        try:
            chapters = await self._chapter_task(
                event_index=index_text,
                session=session,
                on_event=on_event,
                on_token=on_token,
            )
        except Exception:
            logger.warning("Chapter task failed", exc_info=True)
            return

        if not chapters:
            logger.debug("Agent returned no chapters")
            return

        # Validate and convert to ChapterEvents.
        chapters_and_ranges = []
        for ch in chapters:
            if not isinstance(ch, Chapter):
                logger.debug("Skipping non-Chapter object: %s", type(ch).__name__)
                continue
            # 1-based inclusive boundary positions.
            if ch.start < 1 or ch.end < ch.start:
                logger.debug(
                    "Skipping invalid range: start=%d end=%d", ch.start, ch.end
                )
                continue
            if ch.start > len(ranges) or ch.end > len(ranges):
                logger.debug(
                    "Skipping out-of-bounds range: start=%d end=%d (max=%d)",
                    ch.start,
                    ch.end,
                    len(ranges),
                )
                continue

            # Map boundary positions to the underlying log range.
            log_start = ranges[ch.start - 1][0]
            log_end = ranges[ch.end - 1][1]

            chapter_event = ChapterEvent(
                agent_name=self.name,
                name=ch.name,
                message=ch.message,
            )
            chapters_and_ranges.append((log_start, log_end, chapter_event))

        if not chapters_and_ranges:
            return

        try:
            replace_events_with_chapters(state, chapters_and_ranges)
        except ValueError:
            logger.debug("Failed to apply chapters", exc_info=True)
            return

        logger.debug("Applied %d chapter(s)", len(chapters_and_ranges))

        # Emit ChapterEvents so live UIs can update without a reload.
        if on_event is not None:
            for _, _, chapter_event in chapters_and_ranges:
                try:
                    res = on_event(chapter_event)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    pass

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
                    builder = EmissionsBuilder(self.name, exec_state)
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

            except _retryable_exceptions() as e:
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
                    builder = EmissionsBuilder(self.name, exec_state)
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

            except _retryable_exceptions() as e:
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
