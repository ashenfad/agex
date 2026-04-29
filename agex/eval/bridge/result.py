"""
Processes a sandtrap ExecResult into agex's event system.

Each ``python_action`` runs as a fresh script — the namespace is
discarded once execution returns.  ``handle_result`` only reads from
``result``: it fans prints and ``view_image`` outputs into events,
validates the typed return type when ``task_success`` fires, and
re-raises any error captured by the sandbox.  No state is written
back from the namespace.
"""

from __future__ import annotations

import base64
import inspect
import io
from collections.abc import MutableMapping
from typing import Any, Callable

from sandtrap import ExecResult

from agex.agent.datatypes import TaskSuccess
from agex.agent.events import OutputEvent
from agex.eval.bridge.policy import _current_emission_id
from agex.eval.objects import ImageAction, PrintAction
from agex.state.log import add_event_to_log


def handle_result(
    result: ExecResult,
    state: MutableMapping[str, Any],
    agent_name: str,
    on_event: Callable[[Any], None] | None = None,
    emission_id: str | None = None,
) -> None:
    """Process an ExecResult: emit output events, validate task_success,
    re-raise any error.

    Args:
        result: The ExecResult from sandtrap Sandbox.exec().
        state: The kvgit state — used for the event log and return-type
               lookup, never written to from the namespace.
        agent_name: Agent name for event attribution.
        on_event: Optional event callback.
        emission_id: Stamps PrintAction / ImageAction parts so the
                     renderer can pair observations per emission in a
                     multi-emission turn.

    Raises:
        _AgentExit subclasses: TaskSuccess, TaskFail, etc.
        Exception: Any regular exception from agent code.
    """
    # Callers pass ``emission_id`` explicitly.  The contextvar is
    # reset in the caller's ``finally`` block before we run, so
    # reading it here would always return None — falling back to the
    # contextvar is only useful for nested / synthetic callers that
    # didn't know the id up front.
    if emission_id is None:
        emission_id = _current_emission_id.get()

    # 1. Convert print snapshots into a single OutputEvent whose
    #    ``parts`` carries one PrintAction / ImageAction per print
    #    in original order.  Each part stamps the current emission_id
    #    so the renderer can pair observations per emission in a
    #    multi-emission turn.  Intercept __AGEX_IMAGE__: prefixed
    #    prints and convert to ImageAction.
    #
    #    Why one event with N parts instead of N events?  Sandtrap
    #    batches prints into ``result.prints`` regardless — the
    #    sandbox has no per-print callback hook, so emitting an event
    #    per print just fans out a list at the end of exec, not a
    #    stream.  A single multi-part event is fewer commits in
    #    kvgit, one token-budgeting pass, and matches how the
    #    OutputEvent data model already presents itself
    #    (``parts: list[Any]``).
    _IMG_PREFIX = "__AGEX_IMAGE__:"
    output_parts: list[Any] = []
    for args in result.prints:
        tup = tuple(args)
        if len(tup) == 1 and isinstance(tup[0], str) and tup[0].startswith(_IMG_PREFIX):
            try:
                # Defer PIL until we actually need to decode an image —
                # keeps ``import agex`` fast when the agent never prints
                # __AGEX_IMAGE__ markers.
                from PIL import Image  # noqa: PLC0415

                b64 = tup[0][len(_IMG_PREFIX) :]
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                output_parts.append(ImageAction(image=img, emission_id=emission_id))
            except Exception:
                output_parts.append(PrintAction(args=tup, emission_id=emission_id))
        else:
            output_parts.append(PrintAction(args=tup, emission_id=emission_id))
    if output_parts:
        event = OutputEvent(agent_name=agent_name, parts=output_parts)
        add_event_to_log(state, event, on_event=on_event)

    # 2. Convert __outputs__ entries (e.g. view_image) into OutputEvents.
    for item in result.namespace.get("__outputs__", []):
        # Stamp the current emission_id if the ImageAction didn't
        # already carry one (view_image reads the contextvar at call
        # time, so this is just defence in depth).
        if isinstance(item, ImageAction) and item.emission_id is None:
            item.emission_id = emission_id
        event = OutputEvent(agent_name=agent_name, parts=[item])
        add_event_to_log(state, event, on_event=on_event)

    # 3. Validate TaskSuccess result type (moved from sandbox-side closure
    #    so task_success can be a plain picklable function for cross-process)
    if isinstance(result.error, TaskSuccess):
        _validate_task_result(result.error.result, state)

    # 4. Re-raise any error captured by sandtrap.
    # sandtrap catches ALL BaseException (except KeyboardInterrupt) and puts
    # it in result.error. This includes _AgentExit subclasses (TaskSuccess,
    # TaskFail, TaskClarify) which are BaseException.
    if result.error is not None:
        raise result.error


def _validate_task_result(result: Any, state: MutableMapping[str, Any]) -> None:
    """Validate task_success result against the expected return type."""
    return_type = state.get("__expected_return_type__")
    if not return_type or return_type is inspect.Parameter.empty:
        return

    from agex.eval.validation import validate_with_sampling

    try:
        validate_with_sampling(result, return_type)
    except Exception as e:
        if (
            hasattr(return_type, "__module__")
            and hasattr(return_type, "__name__")
            and not hasattr(return_type, "__origin__")
        ):
            type_name = return_type.__name__
        else:
            type_name = str(return_type)
        raise TypeError(
            f"Output validation failed. The returned value did not match "
            f"the expected type '{type_name}'.\nDetails: {e}",
        ) from e
