"""
Bridge layer between agex's registration/state system and sandtrap's Sandbox.

This package translates agex's AgentPolicy and kvgit state into
sandtrap's Policy and namespace dict, then processes the ExecResult
back into agex's state and event system.
"""

import asyncio
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Callable

from sandtrap import sandbox as create_sandbox

from .namespace import build_namespace
from .policy import translate_policy
from .result import handle_result

if TYPE_CHECKING:
    from monkeyfs import FileSystem

    from agex.agent.base import BaseAgent


def _prepare_sandbox(
    program: str,
    agent: "BaseAgent",
    state: MutableMapping[str, Any],
    eval_timeout_seconds: float | None = None,
    *,
    fs: "FileSystem | None" = None,
    on_event: Callable[[Any], None] | None = None,
    file_path: str | None = None,
):
    """Shared setup for execute_sandboxed and aexecute_sandboxed."""
    tick_limit = getattr(agent, "eval_tick_limit", None)

    if tick_limit is not None:
        # Tick limit is the primary protection; use a generous wall-clock
        # safety net so sub-agent LLM calls don't trigger a timeout.
        timeout = 300.0
    else:
        timeout = (
            eval_timeout_seconds
            if eval_timeout_seconds is not None
            else agent.eval_timeout_seconds
        )

    # Ensure IO modules are registered when filesystem is active
    if fs is not None:
        from agex.helpers.stdlib import register_io

        register_io(agent)

    policy = translate_policy(agent, timeout=timeout, tick_limit=tick_limit)
    sb = create_sandbox(
        policy,
        isolation=agent.isolation,
        mode="wrapped",
        filesystem=fs,
        snapshot_prints=True,
    )
    namespace, pre_keys, injected_keys = build_namespace(
        state, agent, agent.name, on_event=on_event
    )

    # Set __file__ for relative import resolution
    if file_path is not None:
        namespace["__file__"] = file_path

    return sb, namespace, pre_keys, injected_keys


def execute_sandboxed(
    program: str,
    agent: "BaseAgent",
    state: MutableMapping[str, Any],
    eval_timeout_seconds: float | None = None,
    *,
    fs: "FileSystem | None" = None,
    session: str = "default",
    on_event: Callable[[Any], None] | None = None,
    on_token: Callable[[Any], None] | None = None,
    file_path: str | None = None,
) -> None:
    """Execute agent code synchronously in the sandtrap sandbox."""
    from .policy import _current_on_event, _current_on_token, _current_session

    sb, namespace, pre_keys, injected_keys = _prepare_sandbox(
        program,
        agent,
        state,
        eval_timeout_seconds,
        fs=fs,
        on_event=on_event,
        file_path=file_path,
    )

    # Set context vars so sub-agent task calls inherit session/on_event/on_token
    session_token = _current_session.set(session)
    event_token = _current_on_event.set(on_event)
    token_token = _current_on_token.set(on_token)
    try:
        if hasattr(sb, "__enter__"):
            with sb:
                result = sb.exec(program, namespace=namespace)
        else:
            result = sb.exec(program, namespace=namespace)
    finally:
        _current_session.reset(session_token)
        _current_on_event.reset(event_token)
        _current_on_token.reset(token_token)

    handle_result(
        result,
        state,
        agent.name,
        pre_keys,
        on_event=on_event,
        injected_keys=injected_keys,
    )


async def aexecute_sandboxed(
    program: str,
    agent: "BaseAgent",
    state: MutableMapping[str, Any],
    eval_timeout_seconds: float | None = None,
    *,
    fs: "FileSystem | None" = None,
    session: str = "default",
    on_event: Callable[[Any], None] | None = None,
    on_token: Callable[[Any], None] | None = None,
    file_path: str | None = None,
) -> None:
    """Execute agent code asynchronously in the sandtrap sandbox.

    Uses sandbox.aexec() so ``await`` works natively in sandbox code.
    """
    from .policy import _current_on_event, _current_on_token, _current_session

    # Wrap async on_event into a thread-safe sync wrapper.  Sandbox code
    # runs in an executor thread, so async
    # callbacks must be dispatched to the event loop.  Fire-and-forget to
    # avoid deadlocks (callback may await things that depend on the sandbox
    # thread continuing).
    safe_on_event = on_event
    if on_event is not None and asyncio.iscoroutinefunction(on_event):
        loop = asyncio.get_running_loop()

        def safe_on_event(event: Any) -> None:
            asyncio.run_coroutine_threadsafe(on_event(event), loop)

    sb, namespace, pre_keys, injected_keys = _prepare_sandbox(
        program,
        agent,
        state,
        eval_timeout_seconds,
        fs=fs,
        on_event=safe_on_event,
        file_path=file_path,
    )

    # Set context vars so sub-agent task calls inherit session/on_event/on_token
    session_token = _current_session.set(session)
    event_token = _current_on_event.set(on_event)
    token_token = _current_on_token.set(on_token)
    try:
        if hasattr(sb, "__enter__"):
            with sb:
                result = await sb.aexec(program, namespace=namespace)
        else:
            result = await sb.aexec(program, namespace=namespace)
    finally:
        _current_session.reset(session_token)
        _current_on_event.reset(event_token)
        _current_on_token.reset(token_token)

    handle_result(
        result,
        state,
        agent.name,
        pre_keys,
        on_event=safe_on_event,
        injected_keys=injected_keys,
    )
