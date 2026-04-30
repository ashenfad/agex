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


def _make_cache_handler(
    state: MutableMapping[str, Any],
) -> Callable[[str, tuple, dict], Any]:
    """Build a sandtrap RPC handler that proxies cache operations to
    a parent-side ``Cache(state)``.

    Used only under process / kernel isolation: the worker's
    ``RemoteCache`` calls ``handler(method, args, kwargs)`` over the
    RPC channel; this factory returns the closure that dispatches by
    method name to the live parent-side cache.

    Errors raised by the parent-side cache (e.g. ``CacheError`` from
    a non-picklable write) propagate back to the worker and re-raise
    in the agent's call site.
    """
    from agex.cache import Cache

    cache = Cache(state)

    def handler(method: str, args: tuple, kwargs: dict) -> Any:
        if method == "getitem":
            return cache[args[0]]
        if method == "setitem":
            cache[args[0]] = args[1]
            return None
        if method == "delitem":
            del cache[args[0]]
            return None
        if method == "iter":
            return list(cache)
        if method == "len":
            return len(cache)
        if method == "contains":
            return args[0] in cache
        raise AttributeError(method)

    return handler


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
    """Shared setup for execute_sandboxed and aexecute_sandboxed.

    Each call builds a fresh namespace.  ``state`` is read only to
    surface ``inputs`` into the namespace; no other keys are read and
    nothing is written.  The caller passes ``state`` to
    :func:`handle_result` for event-log writes.
    """
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

    # Under process / kernel isolation we register an RPC handler so
    # the worker's ``RemoteCache`` proxy can reach back into the
    # parent's live ``Cache(state)``.  Skipped for ``isolation="none"``
    # — the in-process namespace gets the live Cache directly.
    rpc_handlers: dict[str, Callable[[str, tuple, dict], Any]] | None = None
    if agent.isolation != "none":
        rpc_handlers = {"cache": _make_cache_handler(state)}

    sb = create_sandbox(
        policy,
        isolation=agent.isolation,
        mode="wrapped",
        filesystem=fs,
        snapshot_prints=True,
        rpc_handlers=rpc_handlers,
    )
    namespace, _injected_keys = build_namespace(
        state, agent, agent.name, on_event=on_event
    )

    # Set __file__ for relative import resolution
    if file_path is not None:
        namespace["__file__"] = file_path

    return sb, namespace


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
    emission_id: str | None = None,
) -> dict[str, Any]:
    """Execute agent code synchronously in the sandtrap sandbox.

    ``emission_id`` is propagated via contextvar so PrintAction and
    ImageAction parts emitted by this call trace back to the originating
    PythonEmission — essential for per-emission tool_result pairing in
    multi-emission turns.

    Returns the post-exec namespace dict so callers like
    :func:`agex.eval.core.run_file_in_sandbox` can inspect what the
    script computed.  Loop callers driving an LLM ignore this return
    value — namespaces are not shared across emissions.
    """
    from .policy import (
        _current_emission_id,
        _current_on_event,
        _current_on_token,
        _current_parent_log,
        _current_session,
    )

    sb, namespace = _prepare_sandbox(
        program,
        agent,
        state,
        eval_timeout_seconds,
        fs=fs,
        on_event=on_event,
        file_path=file_path,
    )

    # Set context vars so sub-agent task calls inherit session/on_event/on_token
    # and can locate this agent's state+name as their "parent log" target for
    # synthetic OutputEvents carrying sub-agent REPORTs.
    session_token = _current_session.set(session)
    event_token = _current_on_event.set(on_event)
    token_token = _current_on_token.set(on_token)
    parent_log_token = _current_parent_log.set((state, agent.name))
    emission_token = _current_emission_id.set(emission_id)
    try:
        with sb:
            result = sb.exec(program, namespace=namespace)
    finally:
        _current_session.reset(session_token)
        _current_on_event.reset(event_token)
        _current_on_token.reset(token_token)
        _current_parent_log.reset(parent_log_token)
        _current_emission_id.reset(emission_token)

    handle_result(
        result,
        state,
        agent.name,
        on_event=on_event,
        emission_id=emission_id,
    )
    return result.namespace


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
    emission_id: str | None = None,
) -> dict[str, Any]:
    """Execute agent code asynchronously in the sandtrap sandbox.

    Uses sandbox.aexec() so ``await`` works natively in sandbox code.
    Returns the post-exec namespace dict; see :func:`execute_sandboxed`.
    """
    from .policy import (
        _current_emission_id,
        _current_on_event,
        _current_on_token,
        _current_parent_log,
        _current_session,
    )

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

    sb, namespace = _prepare_sandbox(
        program,
        agent,
        state,
        eval_timeout_seconds,
        fs=fs,
        on_event=safe_on_event,
        file_path=file_path,
    )

    # Set context vars so sub-agent task calls inherit session/on_event/on_token
    # and can locate this agent's state+name as their "parent log" target for
    # synthetic OutputEvents carrying sub-agent REPORTs.
    session_token = _current_session.set(session)
    event_token = _current_on_event.set(on_event)
    token_token = _current_on_token.set(on_token)
    parent_log_token = _current_parent_log.set((state, agent.name))
    emission_token = _current_emission_id.set(emission_id)
    try:
        with sb:
            result = await sb.aexec(program, namespace=namespace)
    finally:
        _current_session.reset(session_token)
        _current_on_event.reset(event_token)
        _current_on_token.reset(token_token)
        _current_parent_log.reset(parent_log_token)
        _current_emission_id.reset(emission_token)

    handle_result(
        result,
        state,
        agent.name,
        on_event=safe_on_event,
        emission_id=emission_id,
    )
    return result.namespace
