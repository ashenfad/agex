"""Event-log renderer for the provider-native tool-use wire format.

Converts agex :class:`~agex.agent.events.Event` objects into
provider-agnostic message dicts whose ``content`` uses ``tool_use`` and
``tool_result`` blocks.  Clients translate these dicts to the concrete
shape each provider expects (Anthropic content arrays, OpenAI
``tool_calls`` + ``tool`` role messages, etc.).

Rendering rules:

* Each :class:`ActionEvent` emits one assistant message whose content is
  a list of ``tool_use`` blocks — one per :class:`FileAction` /
  :class:`EditAction` (in order) followed by one for the main
  python_action / terminal_action.
* Each ``tool_use`` needs a matching ``tool_result`` in the next user
  message.  File actions get a synthesized ``"ok"`` result; the main
  action's result is filled by the next :class:`OutputEvent` /
  :class:`SuccessEvent` / :class:`CancelledEvent`.  If none arrives
  (task ended via task_fail/task_clarify without further output), a
  placeholder ``"(no observation)"`` result is synthesized so the
  provider sees a well-formed tool_use → tool_result pairing.
* Non-action events (:class:`TaskStartEvent`, :class:`FileEvent`,
  :class:`SystemNoteEvent`, :class:`ChapterEvent`) are emitted as text
  parts within the surrounding user message to keep role alternation
  clean.
"""

from typing import Any, List

from agex.agent.datatypes import EditAction, FileAction
from agex.agent.events import (
    ActionEvent,
    CancelledEvent,
    ChapterEvent,
    ClarifyEvent,
    ErrorEvent,
    Event,
    FailEvent,
    FileEvent,
    OutputEvent,
    SuccessEvent,
    SystemNoteEvent,
    TaskStartEvent,
)
from agex.llm.core import ContentPart, ImagePart, TextPart
from agex.render.primitives import (
    HI_DETAIL_BUDGET,
    render_chapter,
    render_output_parts_full,
    render_task_start,
)
from agex.render.value import render_value

from .schemas import (
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
    TOOL_TERMINAL,
    TOOL_WRITE_FILE,
)


def _content_parts_to_dicts(parts: List[ContentPart]) -> List[dict]:
    out: List[dict] = []
    for p in parts:
        if isinstance(p, TextPart):
            out.append({"type": "text", "text": p.text})
        elif isinstance(p, ImagePart):
            out.append({"type": "image", "image_data": p.image})
    return out


def _build_action_blocks(
    event: ActionEvent, task_number: int, event_index: int
) -> tuple[list[dict], list[str], str]:
    """Return (tool_use_blocks, file_tool_use_ids, main_tool_use_id)."""
    blocks: list[dict] = []
    file_ids: list[str] = []

    for j, fa in enumerate(event.file_actions):
        block_id = f"toolu_{task_number}_{event_index}_{j}"
        file_ids.append(block_id)
        if isinstance(fa, FileAction):
            file_input: dict[str, Any] = {
                "path": fa.path,
                "content": fa.content,
            }
            if fa.mode != "write":
                file_input["mode"] = fa.mode
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block_id,
                    "name": TOOL_WRITE_FILE,
                    "input": file_input,
                }
            )
        elif isinstance(fa, EditAction):
            edit_input: dict[str, Any] = {"path": fa.path, "search": fa.search}
            if fa.operation == "insert-after":
                edit_input["insert_after"] = fa.content
            elif fa.operation == "insert-before":
                edit_input["insert_before"] = fa.content
            else:
                edit_input["replace"] = fa.content
            if fa.match_all:
                edit_input["match_all"] = True
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block_id,
                    "name": TOOL_EDIT_FILE,
                    "input": edit_input,
                }
            )

    main_id = f"toolu_{task_number}_{event_index}_main"
    if event.terminal:
        main_input: dict[str, Any] = {
            "title": event.title,
            "thinking": event.thinking,
            "commands": event.terminal,
        }
        if getattr(event, "report", ""):
            main_input["report"] = event.report
        blocks.append(
            {
                "type": "tool_use",
                "id": main_id,
                "name": TOOL_TERMINAL,
                "input": main_input,
            }
        )
    else:
        main_input = {
            "title": event.title,
            "thinking": event.thinking,
            "code": event.code or "",
        }
        if getattr(event, "report", ""):
            main_input["report"] = event.report
        blocks.append(
            {
                "type": "tool_use",
                "id": main_id,
                "name": TOOL_PYTHON,
                "input": main_input,
            }
        )

    return blocks, file_ids, main_id


def _file_event_to_text(event: FileEvent) -> str:
    parts: list[str] = []
    if event.added:
        parts.append(f"Added: {', '.join(event.added)}")
    if event.modified:
        parts.append(f"Modified: {', '.join(event.modified)}")
    if event.removed:
        parts.append(f"Removed: {', '.join(event.removed)}")
    return f"[file changes — {event.file_source}] {'; '.join(parts)}"


def _tool_result_block(
    tool_use_id: str,
    content: str | list[dict],
) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


def render_events_as_tool_use(events: List[Event]) -> List[dict]:
    """Render events for the tool-use wire format."""
    messages: List[dict[str, Any]] = []
    # Pending content for the *next* user message.  Mix of text parts
    # and tool_result blocks.  Flushed when the next assistant event
    # arrives or at end of log.
    pending_user: list[dict] = []
    # Tool-use ids from the most recent ActionEvent still awaiting
    # their main tool_result.  None once paired.
    pending_main_id: str | None = None

    task_number = 0
    filtered = [e for e in events if not isinstance(e, ErrorEvent)]

    def flush_user() -> None:
        nonlocal pending_user, pending_main_id
        # Synthesize placeholder main tool_result if none arrived.
        if pending_main_id is not None:
            pending_user.append(_tool_result_block(pending_main_id, "(no observation)"))
            pending_main_id = None
        if pending_user:
            messages.append({"role": "user", "content": pending_user})
            pending_user = []

    for idx, event in enumerate(filtered):
        if isinstance(event, TaskStartEvent):
            task_number += 1
            text, _ = render_task_start(event.message, budget=HI_DETAIL_BUDGET)
            pending_user.append({"type": "text", "text": f"[{task_number}] {text}"})

        elif isinstance(event, ActionEvent):
            # Emit any queued user message first.
            flush_user()
            blocks, file_ids, main_id = _build_action_blocks(event, task_number, idx)
            messages.append({"role": "assistant", "content": blocks})
            # Synthesize file tool_results; defer main.
            for fid in file_ids:
                pending_user.append(_tool_result_block(fid, "ok"))
            pending_main_id = main_id

        elif isinstance(event, OutputEvent):
            parts, _ = render_output_parts_full(event.parts, budget=HI_DETAIL_BUDGET)
            if pending_main_id is not None:
                if parts:
                    has_image = any(isinstance(p, ImagePart) for p in parts)
                    if has_image:
                        content = _content_parts_to_dicts(parts)
                    else:
                        content = "\n".join(
                            p.text for p in parts if isinstance(p, TextPart)
                        )
                else:
                    content = "(no output)"
                pending_user.append(_tool_result_block(pending_main_id, content))
                pending_main_id = None
            # If no pending_main_id (stray OutputEvent before any action),
            # drop — no tool_use to pair with.

        elif isinstance(event, SuccessEvent):
            if pending_main_id is not None:
                estimated = HI_DETAIL_BUDGET * 4
                rendered = render_value(
                    event.result,
                    budget=estimated,
                    token_budget=HI_DETAIL_BUDGET,
                )
                pending_user.append(_tool_result_block(pending_main_id, rendered))
                pending_main_id = None

        elif isinstance(event, CancelledEvent):
            if pending_main_id is not None:
                msg = (
                    f"Task '{event.task_name}' cancelled after "
                    f"{event.iterations_completed} iterations"
                )
                pending_user.append(_tool_result_block(pending_main_id, msg))
                pending_main_id = None

        elif isinstance(event, (FailEvent, ClarifyEvent)):
            # Agent already expressed intent via task_fail/task_clarify in
            # the preceding action's code.  Synthesize a neutral marker
            # so the tool_use pairing is well-formed; the next agent turn
            # sees a new TaskStart anyway.
            if pending_main_id is not None:
                pending_user.append(_tool_result_block(pending_main_id, "(task ended)"))
                pending_main_id = None

        elif isinstance(event, FileEvent):
            pending_user.append({"type": "text", "text": _file_event_to_text(event)})

        elif isinstance(event, ChapterEvent):
            text, _ = render_chapter(event.name, event.message)
            pending_user.append({"type": "text", "text": text})

        elif isinstance(event, SystemNoteEvent):
            pending_user.append({"type": "text", "text": event.message})

    flush_user()
    return messages
